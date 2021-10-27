#pragma once

// Author : Benjamin Lefaudeux (blefaudeux@github)

#include "gaussian_mixture.h"
#include <iostream>
#include <memory>
#include <list>

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
  struct TrackerPerId
  {
    int tracker_id;
    int start_frame;
    std::vector<Target<D>> tracker_per_frame;
    int track_num;

    TrackerPerId(int _tracker_id, int _start_frame)
    : tracker_id(_tracker_id), start_frame(_start_frame), track_num(0)
    {} 
  };

  template <size_t D>
  class IMMGMPHD
  {
    static const size_t S = D * 2;

  public:
    IMMGMPHD()
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

      frame_count = 0;
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
      // m_tgtDynCov = processNoise * processNoise * Matrix<float, S, S>::Identity();
      m_tgtDynCov = processNoise * processNoise * Matrix<float, S, S>::Identity();
      // m_tgtDynCov(0,0) *= pow(m_samplingPeriod, 4) / 4.0;
      // m_tgtDynCov(1,1) *= pow(m_samplingPeriod, 4) / 4.0;
      // m_tgtDynCov(0,2) *= pow(m_samplingPeriod, 3) / 2.0; 
      // m_tgtDynCov(1,3) *= pow(m_samplingPeriod, 3) / 2.0; 
      // m_tgtDynCov(2,0) *= pow(m_samplingPeriod, 3) / 2.0; 
      // m_tgtDynCov(3,1) *= pow(m_samplingPeriod, 3) / 2.0; 
      // m_tgtDynCov(2,2) *= pow(m_samplingPeriod, 2); 
      // m_tgtDynCov(3,3) *= pow(m_samplingPeriod, 2); 
    }

    void setDynamicsModel(MatrixXf const &tgtDynTransitions, MatrixXf const &tgtDynCovariance)
    {
      m_tgtDynTrans = tgtDynTransitions;
      m_tgtDynCov = tgtDynCovariance;
    }

    void setIMMDynamicsModel(float sampling, float dynamicProcessNoise, float staticProcessNoise)
    {
      m_samplingPeriod = sampling;
      m_staticProcessNoise = staticProcessNoise;
      m_DynamicProcessNoise = dynamicProcessNoise;

      // Propagation matrix dynamic
      
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

      // Show trackers
      showTrackerList();
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

    void addTracker(GaussianModel<S> tracker_model)
    {
      int tracker_id = tracker_model.m_track_id;
      Target<D> tracker;
      tracker.id = tracker_id;
      tracker.position[0] = tracker_model.m_mean[0];
      tracker.position[1] = tracker_model.m_mean[1];
      tracker.speed[0] = tracker_model.m_mean[2];
      tracker.speed[1] = tracker_model.m_mean[3];

      auto it = find_if(trackerList.begin(), trackerList.end(), [tracker_id](const TrackerPerId<D> &it)
                        {
          return it.tracker_id == tracker_id;
                        });

      if (it == trackerList.end())
      {
        trackerList.push_back(TrackerPerId<D>(tracker_id, frame_count));
        trackerList.back().tracker_per_frame.push_back(tracker);
      }
      else if (it->tracker_id == tracker_id)
      {
        it->tracker_per_frame.push_back(tracker);
      }
    }

    void showTrackerList()
    {
      for (auto it = trackerList.begin(), it_next = trackerList.begin(); 
              it != trackerList.end(); it = it_next)
      {
        it_next++;
        if (frame_count - (it->start_frame + it->tracker_per_frame.size()) > 20) // not tracked for 2 seconds then erase
        {
          trackerList.erase(it);
        }
        else
        {
          // printf("id: %d\n", it->tracker_id);
          // for (size_t i = 0; i < it->tracker_per_frame.size(); ++i)
          // {
          //   printf("%f %f; ", it->tracker_per_frame[i].speed[0], it->tracker_per_frame[i].speed[1]);
          // }
          // printf("\n");
        }
      }
      ROS_WARN("TrackerList size: %d", (int)trackerList.size());
    }

    void myPruneGaussians(const std::vector<size_t> indices)
    {
      std::map<int, int> matched;
      GaussianMixture<S> pruned_targets;
      pruned_targets.m_gaussians.clear();

      int n_meas = m_measTargets->m_gaussians.size();
      int n_tracker = m_expTargets->m_gaussians.size();

      std::vector<int> i_close_to_best;
      std::vector<bool> merge_checker(indices.size(), false);
      int merge_cnt = 0;

      for (size_t i = 0; i < indices.size(); ++i)
      {
        size_t i_best = indices[i];
        if (i_best == -1 || m_currTargets->m_gaussians[i_best].m_weight < m_pruneTruncThld)
          break;
        
        if (merge_checker[i_best])
        {
          printf("%d already merged\n", i_best);
          // already merged
          continue;
        } 

        GaussianModel<S> merged_model;
        
        i_close_to_best.clear();
        merged_model.clear();
        
        int track_id = m_currTargets->m_gaussians[i_best].m_track_id; 

        // I. Give birth tracker a track id
        if (track_id < 0)
        {
          printf("Birth tracker: -1 -> %d\n", m_ntrack);
          m_currTargets->m_gaussians[i_best].m_track_id = m_ntrack++; 
          track_id = m_currTargets->m_gaussians[i_best].m_track_id;
          merged_model = m_currTargets->m_gaussians[i_best];
        }
        else
        {
          // Find closest gaussians
          TicToc closest_time;
          std::vector<GaussianModel<S>> target_gaussians = m_currTargets->m_gaussians;
          float gauss_distance;
          Matrix<float, D, 1> point_vec;
          Matrix<float, D, 1> mean_vec;
          Matrix<float, D, D> cov;
          i_close_to_best.push_back(i_best);
          merge_checker[i_best] = true;
          merge_cnt++;
          for (size_t j = 0; j < target_gaussians.size(); ++j)
          {
            if (j != i_best && !merge_checker[j])
            {
              // compute distance
              point_vec = target_gaussians[j].m_mean.head(D);
              mean_vec =  target_gaussians[i_best].m_mean.head(D);
              cov = target_gaussians[i_best].m_cov.topLeftCorner(D, D);
              gauss_distance = mahalanobis<2>(point_vec, mean_vec, cov);
              // printf("pt: %f, %f; mean: %f, %f, cov: %f %f %f %f, dist: %f\n", point_vec(0), point_vec(1), mean_vec(0), mean_vec(1), cov(0,0), cov(0,1), cov(1,0), cov(1,1), gauss_distance);
              if ((gauss_distance < m_pruneMergeThld) && (target_gaussians[j].m_weight != 0.0))
              {
                merge_checker[j] = true;
                merge_cnt++;
                // printf("compared id: %d; %f; %f %f %f; dist: %f\n", target_gaussians[j].m_track_id, target_gaussians[j].m_weight, target_gaussians[j].m_mean[0], target_gaussians[j].m_mean[1], 0.0, gauss_distance);
                i_close_to_best.push_back(j);
              }
            }
          }
          printf("curr tracker: %d; %f %f %f ;size: %d\n", track_id, target_gaussians[i_best].m_mean[0], target_gaussians[i_best].m_mean[1], 0.0, (int)i_close_to_best.size());
          ROS_WARN("Find closest: %f ms\n", closest_time.toc());
          // merge gaussians
          TicToc merge_time;

          if (i_close_to_best.size() > 1)
          {
            merged_model.m_track_id = target_gaussians[i_best].m_track_id;

            for (auto const &idx : i_close_to_best)
            {
              merged_model.m_weight += target_gaussians[idx].m_weight;
            }

            for (auto const &idx : i_close_to_best)
            {
              merged_model.m_mean += target_gaussians[idx].m_mean * target_gaussians[idx].m_weight;
            }

            if (merged_model.m_weight != 0.f)
              merged_model.m_mean /= merged_model.m_weight;
            
            merged_model.m_cov.setZero();
            for(auto const &idx : i_close_to_best)
            {
              Matrix<float, S, 1> diff = merged_model.m_mean - target_gaussians[idx].m_mean;
              merged_model.m_cov += target_gaussians[idx].m_weight * (target_gaussians[idx].m_cov + diff * diff.transpose());
            }

            if (merged_model.m_weight != 0.f)
              merged_model.m_cov /= merged_model.m_weight;
          }
          else
          {
            merged_model = target_gaussians[i_close_to_best[0]];
          }
          ROS_WARN("Merge time: %f ms\n", merge_time.toc());
        }

        // II. Add survived tracker
        if (i_best < n_tracker) // survived filter (propagated) + birth + survived_filter(not propagated)
        {
          printf("Survived tracker: %d %f; %f %f 0.0\n", track_id, merged_model.m_weight, merged_model.m_mean[0], merged_model.m_mean[1]);
          if (matched.find(track_id) == matched.end())
          {
            printf("Added\n");
            matched.insert(std::make_pair(track_id, -1));
            addTracker(merged_model);
            pruned_targets.m_gaussians.emplace_back(std::move(merged_model));
          }
        }
        // III. Add new trackers
        else 
        {
          size_t i_meas = (i_best - n_tracker) / n_tracker;
          int meas_id = m_measTargets->m_gaussians[i_meas].m_track_id;
          printf("New track_id: %d, meas_id: %d, track weight: %f; %f %f 0.0\n", track_id, meas_id, merged_model.m_weight, merged_model.m_mean[0], merged_model.m_mean[1]);
          if (matched.find(track_id) == matched.end()) // new tracker
          {
            printf("Added\n");
            addTracker(merged_model);
            matched.insert(std::make_pair(track_id, meas_id));
            pruned_targets.m_gaussians.emplace_back(std::move(merged_model));  
          }
        }

        if (pruned_targets.m_gaussians.size() > m_nMaxPrune) // max tracker
          break;
      }
      ROS_WARN("%d out of %d has merged", merge_cnt, (int)indices.size());
      
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
          // printf("Meas id: %d, %f;%f;0.0   birth gaussian: %d, %f; %f; 0.0   weight: %f\n", 
          //   meas_id, m_measTargets->m_gaussians[k].m_mean[0], m_measTargets->m_gaussians[k].m_mean[1],
          //   birth_gaussian.m_track_id, birth_gaussian.m_mean[0], birth_gaussian.m_mean[1], birth_gaussian.m_weight);
          birth_gaussian.m_track_id = m_ntrack++;
          birth_gaussian.m_weight = birth_gaussian.m_weight < 0.2f ? 0.2f: birth_gaussian.m_weight;
          pruned_targets.m_gaussians.emplace_back(std::move(birth_gaussian));
        }
      }
      m_currTargets->m_gaussians = pruned_targets.m_gaussians;
      frame_count++;
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
        // Compute the expected measurement
        m_expMeasure.push_back(m_obsMat * tgt.m_mean);
        m_expDisp.push_back(m_obsCov + m_obsMat * tgt.m_cov * m_obsMatT);
        m_uncertainty.push_back(tgt.m_cov * m_obsMatT * m_expDisp.back().inverse());
        m_covariance.push_back((Matrix<float, S, S>::Identity() - m_uncertainty.back() * m_obsMat) * tgt.m_cov);
        printf("Expected: %f;%f; %f;%f; vel: %f; %f; id: %d\n", m_expMeasure.back()[0], m_expMeasure.back()[1], m_covariance.back()(0,0), m_covariance.back()(1,1), m_uncertainty.back()(0,0), m_uncertainty.back()(1,1), tgt.m_track_id);
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
          // printf("exp target: %f;%f;%f;%f\n", m_expTargets->m_gaussians[n_targt].m_mean[0], m_expTargets->m_gaussians[n_targt].m_mean[1], m_expTargets->m_gaussians[n_targt].m_mean[2],m_expTargets->m_gaussians[n_targt].m_mean[3]);
          // printf("meas mean: %f;%f;%f;%f, expMeasure: %f;%f;%f;%f\n", measuredTarget.m_mean[0], measuredTarget.m_mean[1], measuredTarget.m_mean[2], measuredTarget.m_mean[3], m_expMeasure[n_targt][0], m_expMeasure[n_targt][1], m_expMeasure[n_targt][2], m_expMeasure[n_targt][3]);
          // printf("diff: %f; %f; %f; %f\n", measuredTarget.m_mean[0] - m_expMeasure[n_targt][0], measuredTarget.m_mean[1]-m_expMeasure[n_targt][1], measuredTarget.m_mean[2]-m_expMeasure[n_targt][2], measuredTarget.m_mean[3]-m_expMeasure[n_targt][3]);
          // std::cout << "cov: " << std::endl;
          // std::cout << m_uncertainty[n_targt] <<std::endl;
          // printf("matched mean: %f;%f;%f;%f\n", matchTarget.m_mean[0], matchTarget.m_mean[1], matchTarget.m_mean[2], matchTarget.m_mean[3]);
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
    float m_staticProcessNoise;
    float m_DynamicProcessNoise;

    float m_pruneMergeThld;
    float m_pruneTruncThld;

    float m_measNoisePose;
    float m_measNoiseSpeed;
    float m_measNoiseBackground; // Background detection "noise", other models are possible..

    std::vector<int> m_iBirthTargets;

    Matrix<float, S, S> m_tgtDynTrans;
    Matrix<float, S, S> m_tgtDynCov;

    Matrix<float, S, S> m_dynTrans;
    Matrix<float, S, S> m_dynCov;
    Matrix<float, S, S> m_staTrans;
    Matrix<float, S, S> m_staCov;

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

    int frame_count;
    list<TrackerPerId<D>> trackerList;
  };
}