#pragma once

// Author : Benjamin Lefaudeux (blefaudeux@github)

#include "gaussian_mixture.h"
#include <iostream>
#include <memory>

namespace gmphd
{
  using namespace Eigen;

  template <size_t S>
  struct SpawningModel
  {
    SpawningModel()
    {
      m_trans.setIdentity();
      m_cov.setIdentity();
      m_offset.setZero();
      m_weight = 0.9f;
    }

    float m_weight;

    Matrix<float, S, S> m_trans;
    Matrix<float, S, S> m_cov;
    Matrix<float, S, 1> m_offset;
  };

  template <size_t D>
  struct Target
  {
    Matrix<float, D, 1> position;
    Matrix<float, D, 1> speed;
    float weight;

    int id;
  };

  template <size_t D>
  class GMPHD
  {
    static const size_t S = D * 2;

  public:
    GMPHD()
    {
      m_ntrack = 0;
      m_pruneTruncThld = 0.f;
      m_pDetection = 0.f;
      m_pSurvival = 0.f;

      // Initialize all gaussian mixtures, we know the dimension now
      m_measTargets.reset(new GaussianMixture<S>());
      m_birthTargets.reset(new GaussianMixture<S>());
      m_currTargets.reset(new GaussianMixture<S>());
      m_expTargets.reset(new GaussianMixture<S>());
      m_extractedTargets.reset(new GaussianMixture<S>());
      m_spawnTargets.reset(new GaussianMixture<S>());
      m_tempTargets.reset(new GaussianMixture<S>());
    }

    // Input: raw measurements and possible ref change
    void setNewReferential(Matrix4f const &transform)
    {
      // Change referential for every gaussian in the gaussian mixture
      m_currTargets->changeReferential(transform);
    }

    void setNewMeasurements(std::vector<Target<D>> const &measurements)
    {
      // Clear the gaussian mixture
      m_measTargets->m_gaussians.clear();

      for (const auto &meas : measurements)
      {
        // Create new gaussian model according to measurement
        GaussianModel<S> new_obs;
        new_obs.m_mean.template head<D>() = meas.position;
        new_obs.m_mean.template tail<D>() = meas.speed;
        new_obs.m_cov = m_obsCov;
        new_obs.m_weight = meas.weight;
        new_obs.m_track_id = meas.id;

        m_measTargets->m_gaussians.push_back(std::move(new_obs));
      }
    }

    // Output
    std::vector<Target<D>> getTrackedTargets(float const &extract_thld)
    {
      // Get through every target, keep the ones whose weight is above threshold
      float const thld = std::max(extract_thld, 0.f);
      m_extractedTargets->m_gaussians.clear();
      std::copy_if(begin(m_currTargets->m_gaussians), end(m_currTargets->m_gaussians), std::back_inserter(m_extractedTargets->m_gaussians), [&thld](const GaussianModel<S> &gaussian) { return gaussian.m_weight >= thld; });

      // Fill in "extracted_targets" from the "current_targets"
      std::vector<Target<D>> targets;
      for (auto const &gaussian : m_extractedTargets->m_gaussians)
      {
        targets.push_back({.position = gaussian.m_mean.template head<D>(), .speed = gaussian.m_mean.template tail<D>(), .weight = gaussian.m_weight, .id = gaussian.m_track_id});
      }
      return targets;
    }

    // Parameters to set before use
    void setDynamicsModel(float sampling, float processNoise)
    {
      m_samplingPeriod = sampling;
      m_processNoise = processNoise;

      // Fill in propagation matrix :
      m_tgtDynTrans.setIdentity();

      for (uint i = 0; i < D; ++i)
      {
        m_tgtDynTrans(i, D + i) = m_samplingPeriod;
      }

      // Fill in covariance matrix
      // Extra covariance added by the dynamics. Could be 0.
      m_tgtDynCov = processNoise * processNoise * Matrix<float, S, S>::Identity();
    }

    void setDynamicsModel(MatrixXf const &tgtDynTransitions, MatrixXf const &tgtDynCovariance)
    {
      m_tgtDynTrans = tgtDynTransitions;
      m_tgtDynCov = tgtDynCovariance;
    }

    void setSurvivalProbability(float prob_survival)
    {
      m_pSurvival = prob_survival;
    }

    void setObservationModel(float probDetectionOverall, float measNoisePose,
                             float measNoiseSpeed, float measNoiseBackground)
    {
      m_pDetection = probDetectionOverall;
      m_measNoisePose = measNoisePose;
      m_measNoiseSpeed = measNoiseSpeed;
      m_measNoiseBackground = measNoiseBackground; // False detection probability

      // Set model matrices
      m_obsMat.setIdentity();
      m_obsMatT = m_obsMat.transpose();
      m_obsCov.setIdentity();

      m_obsCov.template topLeftCorner<D, D>() *= m_measNoisePose * m_measNoisePose;
      m_obsCov.template bottomRightCorner<D, D>() *= m_measNoiseSpeed * m_measNoiseSpeed;
    }

    void setPruningParameters(float prune_trunc_thld, float prune_merge_thld,
                              int prune_max_nb)
    {
      m_pruneTruncThld = prune_trunc_thld;
      m_pruneMergeThld = prune_merge_thld;
      m_nMaxPrune = prune_max_nb;
    }

    void setBirthModel(const GaussianMixture<S> &birthModel)
    {
      m_birthModel.reset(new GaussianMixture<S>(birthModel));

      // Mark the targets as "false", in that they do not match any measure really
      for (auto &gaussian : m_birthModel->m_gaussians)
      {
        gaussian.m_isFalseTarget = true;
      }
    }

    void setSpawnModel(std::vector<SpawningModel<S>> &spawnModels)
    {
      m_spawnModels = spawnModels;
    }

    void propagate()
    {
      m_nPredTargets = 0;

      // Predict new targets (spawns):
      predictBirth();

      // Predict propagation of expected targets
      predictTargets();

      // Build the update components
      buildUpdate();

      // Update the probabilities
      update();

      // sort by decreasing weights
      size_t target_num = m_currTargets->m_gaussians.size();
      std::vector<GaussianModel<S>> gaussians = m_currTargets->m_gaussians;
      std::vector<size_t> ordered_indices(target_num);
      std::iota(ordered_indices.begin(), ordered_indices.end(), 0);
      std::stable_sort(ordered_indices.begin(), ordered_indices.end(), 
        [&gaussians](size_t i1, size_t i2){return gaussians[i1].m_weight > gaussians[i2].m_weight;});

      // Association
      // associate(ordered_indices);

      myPruneGaussians(ordered_indices);

      // Prune gaussians (remove weakest, merge close enough gaussians)
      // pruneGaussians();

      // Clean std::vectors :
      m_expMeasure.clear();
      m_expDisp.clear();
      m_uncertainty.clear();
      m_covariance.clear();
    }

    void reset()
    {
      m_currTargets->m_gaussians.clear();
      m_extractedTargets->m_gaussians.clear();
    }

  private:
    /*!
     * \brief The spawning models (how gaussians spawn from existing targets)
     * Example : how airplanes take off from a carrier..
     */
    std::vector<SpawningModel<S>> m_spawnModels;

    void myPruneGaussians(const std::vector<size_t> indices)
    {
      std::map<int, int> matched;
      GaussianMixture<S> pruned_targets;
      pruned_targets.m_gaussians.clear();

      int n_meas = m_measTargets->m_gaussians.size();
      int n_tracker = m_expTargets->m_gaussians.size();
      for (size_t i = 0; i < indices.size(); ++i)
      {
        size_t i_best = indices[i];
        if (i_best == -1 || m_currTargets->m_gaussians[i_best].m_weight < 0.1)
          break;
        
        int track_id = m_currTargets->m_gaussians[i_best].m_track_id; 
        // I. Give birth tracker a track id
        if (track_id < 0)
        {
          printf("Birth tracker: -1 -> %d\n", m_ntrack);
          m_currTargets->m_gaussians[i_best].m_track_id = m_ntrack++; 
          track_id = m_currTargets->m_gaussians[i_best].m_track_id;
        }

        // II. Add survived tracker
        if (i_best < n_tracker) // survived filter (propagated) + birth + survived_filter(not propagated)
        {
          printf("Survived tracker: %d %f\n", track_id, m_currTargets->m_gaussians[i_best].m_weight);
          if (matched.find(track_id) == matched.end())
          {
            printf("Added\n");
            matched.insert(std::make_pair(track_id, -1));
            pruned_targets.m_gaussians.emplace_back(std::move(m_currTargets->m_gaussians[i_best]));
          }
        }
        // III. Add new trackers
        else 
        {
          size_t i_meas = (i_best - n_tracker) / n_tracker;
          int meas_id = m_measTargets->m_gaussians[i_meas].m_track_id;
          printf("New track_id: %d, meas_id: %d, n_meas: %d, n_trk: %d\n", track_id, meas_id, n_meas, n_tracker);
          if (matched.find(track_id) == matched.end()) // new tracker
          {
            printf("Added\n");
            matched.insert(std::make_pair(track_id, meas_id));
            pruned_targets.m_gaussians.emplace_back(std::move(m_currTargets->m_gaussians[i_best]));  
          }
        }
        if (pruned_targets.m_gaussians.size() > 150) // max tracker
          break;
      }
      
      // IV. For measurements that are not associated, add new tracker
      size_t birth_idx = m_nPredTargets - m_spawnTargets->m_gaussians.size() - m_birthTargets->m_gaussians.size();
      for (size_t k = 0; k < m_measTargets->m_gaussians.size(); ++k)
      {
        int meas_id = m_measTargets->m_gaussians[k].m_track_id;
        auto result = std::find_if(matched.begin(), matched.end(), 
          [meas_id](const auto &element){return element.second == meas_id;});
        if (result == matched.end()) // not matched
        {
          size_t gauss_idx = m_nPredTargets * (k+1) + birth_idx - 1;
          // printf("Birth idx: %d, Gauss_idx: %d, exp size: %d\n", birth_idx, gauss_idx, m_expTargets->m_gaussians.size());
          GaussianModel<S> birth_gaussian = m_currTargets->m_gaussians[gauss_idx];
          printf("Meas id: %d, %f;%f;0.0   birth gaussian: %d, %f; %f; 0.0   weight: %f\n", 
            meas_id, m_measTargets->m_gaussians[k].m_mean[0], m_measTargets->m_gaussians[k].m_mean[1],
            birth_gaussian.m_track_id, birth_gaussian.m_mean[0], birth_gaussian.m_mean[0], birth_gaussian.m_weight);
          birth_gaussian.m_track_id = m_ntrack++;
          birth_gaussian.m_weight = 0.2f;
          pruned_targets.m_gaussians.emplace_back(std::move(birth_gaussian));
        }
      }
      m_currTargets->m_gaussians = pruned_targets.m_gaussians;
    }

    void associate(const std::vector<size_t> indices)
    {
      int meas_size = m_measTargets->m_gaussians.size();
      int state_size = m_expTargets->m_gaussians.size();

      std::vector<std::pair<int, int>> matched;
      std::vector<int> matched_trackers;

      for (size_t i = 0; i < indices.size(); ++i)
      {
        size_t best_idx = indices[i];
        size_t tracker_idx = best_idx % state_size;
        int track_id = m_currTargets->m_gaussians[best_idx].m_track_id;
        if (best_idx < state_size) // from original spawn from previous frame
        {
          printf("Spawned from previous but best: %d\n", track_id);
        } 
        else
        {
          int meas_idx = (best_idx - state_size) / state_size;
          int meas_id = m_measTargets->m_gaussians[meas_idx].m_track_id;
          printf("meas_idx: %d, best_idex: %d, track_id: %d, meas_id: %d, weight: %f\n", meas_idx, best_idx, track_id, meas_id, m_currTargets->m_gaussians[best_idx].m_weight);
          auto find_tracker = std::find_if(matched.begin(), matched.end(), 
                                          [&tracker_idx](const std::pair<int, int>& element){return element.first == tracker_idx;});
          auto find_meas = std::find_if(matched.begin(), matched.end(), 
                                          [&meas_idx](const std::pair<int, int>& element){return element.second == meas_idx;});
          
          if (find_tracker == matched.end() && find_meas == matched.end()) // new tracker & new meas
          {
            matched.push_back(std::make_pair(tracker_idx, meas_idx));
          }

          if (track_id < 0) // first seen (birth)
          {
            m_currTargets->m_gaussians[best_idx].m_track_id = meas_id;
          }
        }
      }

      std::vector<int> unmatched_trackers;
      std::vector<int> unmatched_meas;
      printf("unmatched det: ");
      for (size_t j = 0; j < m_measTargets->m_gaussians.size(); ++j)
      {
        auto result = std::find_if(matched.begin(), matched.end(), 
                                  [&j](const std::pair<int, int>& element){return element.second == j;});
        if (result == matched.end())
        {
          unmatched_meas.push_back(j);
          printf("%d ", j);
        }
      }
      printf("\n");
      printf("unmatched trk:");
      for (size_t j = 0; j < m_expTargets->m_gaussians.size(); ++j)
      {
        auto result = std::find_if(matched.begin(), matched.end(), 
                                  [&j](const std::pair<int, int>& element){return element.first == j;});
        if (result == matched.end())
        {
          printf("%d ", j);
          unmatched_trackers.push_back(j);
        }
      }
      printf("\n");
    }

    void buildUpdate()
    {

      // Concatenate all the wannabe targets :
      // - birth targets
      if (m_birthTargets->m_gaussians.size() > 0)
      {
        m_iBirthTargets.resize(m_birthTargets->m_gaussians.size());
        std::iota(m_iBirthTargets.begin(), m_iBirthTargets.end(), m_birthTargets->m_gaussians.size());
        m_expTargets->m_gaussians.insert(m_expTargets->m_gaussians.end(), m_birthTargets->m_gaussians.begin(),
                                         m_birthTargets->m_gaussians.begin() + m_birthTargets->m_gaussians.size());
      }

      // - spawned targets
      if (m_spawnTargets->m_gaussians.size() > 0)
      {
        m_expTargets->m_gaussians.insert(m_expTargets->m_gaussians.end(), m_spawnTargets->m_gaussians.begin(),
                                         m_spawnTargets->m_gaussians.begin() + m_spawnTargets->m_gaussians.size());
      }

      // Compute PHD update components (for every expected target)
      m_nPredTargets = m_expTargets->m_gaussians.size();
      printf("Number of predicted targets: %d\n", m_nPredTargets);
      m_expMeasure.clear();
      m_expMeasure.reserve(m_nPredTargets);

      m_expDisp.clear();
      m_expDisp.reserve(m_nPredTargets);

      m_uncertainty.clear();
      m_uncertainty.reserve(m_nPredTargets);

      m_covariance.clear();
      m_covariance.reserve(m_nPredTargets);

      for (auto const &tgt : m_expTargets->m_gaussians)
      {
        printf("Expected: %f;%f;0.0 id: %d\n", tgt.m_mean[0], tgt.m_mean[1], tgt.m_track_id);
        // Compute the expected measurement
        m_expMeasure.push_back(m_obsMat * tgt.m_mean);
        m_expDisp.push_back(m_obsCov + m_obsMat * tgt.m_cov * m_obsMatT);

        m_uncertainty.push_back(tgt.m_cov * m_obsMatT * m_expDisp.back().inverse());
        m_covariance.push_back((Matrix<float, S, S>::Identity() - m_uncertainty.back() * m_obsMat) * tgt.m_cov);
      }
    }

    void predictBirth()
    {
      m_spawnTargets->m_gaussians.clear();
      m_birthTargets->m_gaussians.clear();

      // -----------------------------------------
      // Compute spontaneous births
      m_birthTargets->m_gaussians = m_birthModel->m_gaussians;

      m_nPredTargets += m_birthTargets->m_gaussians.size();

      // -----------------------------------------
      // Compute spawned targets
      printf("Spawn: ");
      for (auto const &curr : m_currTargets->m_gaussians)
      {
        for (auto const &spawn : m_spawnModels)
        {
          GaussianModel<S> new_spawn;

          // Define a gaussian model from the existing target
          // and spawning properties
          new_spawn.m_weight = curr.m_weight * spawn.m_weight;
          new_spawn.m_mean = spawn.m_offset + spawn.m_trans * curr.m_mean;
          new_spawn.m_cov = spawn.m_cov + spawn.m_trans * curr.m_cov * spawn.m_trans.transpose();
          new_spawn.m_isFalseTarget = true;
          printf("%d ", curr.m_track_id);
          new_spawn.m_track_id = curr.m_track_id;
          // Add this new gaussian to the list of expected targets
          m_spawnTargets->m_gaussians.push_back(std::move(new_spawn));

          // Update the number of expected targets
          ++m_nPredTargets;
        }
      }
      printf("\n");
    }

    void predictTargets()
    {
      m_expTargets->m_gaussians.clear();
      m_expTargets->m_gaussians.reserve(m_currTargets->m_gaussians.size());
      printf("Targets ");
      for (auto const &curr : m_currTargets->m_gaussians)
      {
        // Compute the new shape of the target
        GaussianModel<S> new_target;
        new_target.m_weight = m_pSurvival * curr.m_weight;
        new_target.m_mean = m_tgtDynTrans * curr.m_mean;
        new_target.m_cov = m_tgtDynCov + m_tgtDynTrans * curr.m_cov * m_tgtDynTrans.transpose();
        new_target.m_track_id = curr.m_track_id;
        printf("%d ", curr.m_track_id);
        // Push back to the expected targets
        m_expTargets->m_gaussians.push_back(new_target);
        ++m_nPredTargets;
      }
      printf("\n");
    }

    void pruneGaussians()
    {
      m_currTargets->prune(m_pruneTruncThld, m_pruneMergeThld, m_nMaxPrune);
    }

    void update()
    {
      m_currTargets->m_gaussians.clear();

      // We'll consider every possible association : std::vector size is (expected targets)*(measured targets)
      m_currTargets->m_gaussians.reserve((m_measTargets->m_gaussians.size() + 1) *
                                         m_expTargets->m_gaussians.size());

      printf("Update - meas %d, exp: %d\n", (int)m_measTargets->m_gaussians.size(), (int)m_expTargets->m_gaussians.size());

      // First set of gaussians : mere propagation of existing ones
      // don't propagate the "birth" targets... we set their weight to 0
      for (auto const &target : m_expTargets->m_gaussians)
      {
        // Copy the target into the final set, adjust the weight if it was spawned
        auto newTarget = target;
        // newTarget.m_weight = target.m_isFalseTarget ? 0.f : (1.f - m_pDetection) * target.m_weight;
        newTarget.m_weight = target.m_isFalseTarget ? 0.f : target.m_weight;
        newTarget.m_track_id = target.m_track_id;
        printf("new target: %d, %f\n", newTarget.m_track_id, newTarget.m_weight);
        m_currTargets->m_gaussians.emplace_back(std::move(newTarget));
      }

      uint cur_id = 0;
      std::vector<int> best_indices;
      // Second set of gaussians : match observations and previsions
      for (auto &measuredTarget : m_measTargets->m_gaussians)
      {
        m_tempTargets->m_gaussians.clear();
        uint start_normalize = m_currTargets->m_gaussians.size();

        for (uint n_targt = 0; n_targt < m_expTargets->m_gaussians.size(); ++n_targt)
        {

          // Compute matching factor between predictions and measures.
          const auto distance = mahalanobis<2>(measuredTarget.m_mean.template head<D>(),
                                               m_expMeasure[n_targt].template head<D>(),
                                               m_expDisp[n_targt].template topLeftCorner<D, D>());
          GaussianModel<S> matchTarget;

          matchTarget.m_weight = m_pDetection * m_expTargets->m_gaussians[n_targt].m_weight / distance;

          matchTarget.m_mean = m_expTargets->m_gaussians[n_targt].m_mean +
                               m_uncertainty[n_targt] * (measuredTarget.m_mean - m_expMeasure[n_targt]);

          matchTarget.m_cov = m_covariance[n_targt];
          // matchTarget.m_track_id = m_ntrack++;
          matchTarget.m_track_id = m_expTargets->m_gaussians[n_targt].m_track_id;
          // printf("meas mean: %f;%f, expMeasure: %f;%f, expDisp: %f;%f\n", measuredTarget.m_mean[0], measuredTarget.m_mean[1], m_expMeasure[n_targt][0], m_expMeasure[n_targt][1], m_expDisp[n_targt](0, 0), m_expDisp[n_targt](1,1));
          // printf("distance: %f, weight: %f, mean: %f;%f\n", distance, m_expTargets->m_gaussians[n_targt].m_weight, matchTarget.m_mean[0], matchTarget.m_mean[1]);

          m_currTargets->m_gaussians.emplace_back(std::move(matchTarget));
        }
        cur_id++;
        // Normalize weights in the same predicted set, taking clutter into account
        m_currTargets->normalize(m_measNoiseBackground, start_normalize,
                                 m_currTargets->m_gaussians.size(), 1);
      }
      printf("CurrTargets num: %d\n", (int)m_currTargets->m_gaussians.size());
      // print
      printf("Unassociated\n");
      int meas_idx = -1;
      for (size_t i = 0; i < m_currTargets->m_gaussians.size(); ++i)
      {
        if (m_currTargets->m_gaussians[i].m_track_id < 0)
        {
          if (meas_idx < 0){
            meas_idx++;
            continue;
          } else {
            printf("meas_id: %d, loc: %f;%f;0.0, weight: %f\n", 
              m_measTargets->m_gaussians[meas_idx].m_track_id, 
              m_measTargets->m_gaussians[meas_idx].m_mean[0],
              m_measTargets->m_gaussians[meas_idx].m_mean[1],
              m_currTargets->m_gaussians[i].m_weight);
            meas_idx++;
          }            
        }
      }
    }

  private:
    bool m_motionModel;

    uint m_maxGaussians;
    uint m_nPredTargets;
    uint m_nCurrentTargets;
    uint m_nMaxPrune;
    uint m_ntrack;


    float m_pSurvival;
    float m_pDetection;

    float m_samplingPeriod;
    float m_processNoise;

    float m_pruneMergeThld;
    float m_pruneTruncThld;

    float m_measNoisePose;
    float m_measNoiseSpeed;
    float m_measNoiseBackground; // Background detection "noise", other models are possible..

    std::vector<int> m_iBirthTargets;

    Matrix<float, S, S> m_tgtDynTrans;
    Matrix<float, S, S> m_tgtDynCov;

    Matrix<float, S, S> m_obsMat;
    Matrix<float, S, S> m_obsMatT;
    Matrix<float, S, S> m_obsCov;

    // Temporary matrices, used for the update process
    std::vector<Matrix<float, S, S>> m_covariance;
    std::vector<Matrix<float, S, 1>> m_expMeasure;
    std::vector<Matrix<float, S, S>> m_expDisp;
    std::vector<Matrix<float, S, S>> m_uncertainty;

    std::unique_ptr<GaussianMixture<S>> m_birthModel;

    std::unique_ptr<GaussianMixture<S>> m_birthTargets;
    std::unique_ptr<GaussianMixture<S>> m_currTargets;
    std::unique_ptr<GaussianMixture<S>> m_expTargets;
    std::unique_ptr<GaussianMixture<S>> m_extractedTargets;
    std::unique_ptr<GaussianMixture<S>> m_measTargets;
    std::unique_ptr<GaussianMixture<S>> m_spawnTargets;
    std::unique_ptr<GaussianMixture<S>> m_tempTargets;
  };
}