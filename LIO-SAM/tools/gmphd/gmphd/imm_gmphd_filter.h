#pragma once

// Author : Benjamin Lefaudeux (blefaudeux@github)

#include "gaussian_mixture_model.h"
#include <iostream>
#include <memory>
#include <list>

bool tracking_debug = false;

namespace gmphd
{
  using namespace Eigen;

  template <size_t S>
  struct SpawningModel
  {
    SpawningModel()
    {
      m_trans.setIdentity();
      m_trans(2,2) = 0.0;
      m_trans(3,3) = 0.0;
      m_cov.setIdentity();
      m_offset.setZero();
      m_weight = 1.0f;
      // m_cov *= 120;
      m_cov *= 1e-24;
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
    int type;
    float prob;
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
    static const size_t M = 2;
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

      associated.clear();

      // IMM
      frame_count = 0;
      m_mode_trans << 0.9, 0.1, 0.1, 0.9; // p(0->0 = 0.9) + p(0->1 = 0.1) = 1 ; p(1->0 = 0.1) + p(1->1 = 0.9) = 1
      
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

    std::vector<Target<D>> getTrackedTargets2(float const &extract_thld)
    {
      // Get through every target, keep the ones whose weight is above threshold
      float const thld = std::max(extract_thld, 0.f);
      m_extractedTargets->m_gaussians.clear();
      std::copy_if(begin(m_currTargets->m_fusedGaussians), end(m_currTargets->m_fusedGaussians), std::back_inserter(m_extractedTargets->m_gaussians), [&thld](const GaussianModel<S> &gaussian) { return gaussian.m_weight >= thld; });

      // Fill in "extracted_targets" from the "current_targets"
      std::vector<Target<D>> targets;
      for (auto const &gaussian : m_extractedTargets->m_gaussians)
      {
        targets.push_back({.position = gaussian.m_mean.template head<D>(), .speed = gaussian.m_mean.template tail<D>(), .weight = gaussian.m_weight, .id = gaussian.m_track_id, .type = gaussian.model_type, .prob = gaussian.model_prob,});
      }
      return targets;
    }

    std::map<int, int> getAssociationMap() {
      return associated;
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

    void setStaticModel(float sampling, float processNoise)
    {
      m_tgtStaTrans.setIdentity();
      m_tgtStaCov = processNoise * processNoise * Matrix<float, S, S>::Identity();
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

    void setObservationModel2(float probDetectionOverall, float measNoisePose, float measNoiseBackground){
      m_pDetection = probDetectionOverall;
      m_measNoisePose = measNoisePose;
      m_measNoiseBackground = measNoiseBackground;
      m_obsMat2 = Matrix<float, D, S>::Zero();
      m_obsMat2(0,0) = 1.0;
      m_obsMat2(1,1) = 1.0;
      m_obsMat2T = m_obsMat2.transpose();
      m_obsCov2.setIdentity();
      m_obsCov2 *= m_measNoisePose * m_measNoisePose;
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

    void stateInteraction()
    {
      for (size_t k = 0; k < m_currTargets->m_gaussians.size(); ++k)
      {
        VectorXf C = VectorXf::Zero(M); // c 
        VectorXf Pr = VectorXf::Zero(M); // model_probability
        MatrixXf U = MatrixXf::Zero(M, M);
        MatrixXf X = MatrixXf::Zero(S, M);
        Pr(0) = m_currTargets->m_gaussians[k].model_prob;
        Pr(1) = m_currTargets->m_gaussians2[k].model_prob;  
        X.col(0) = m_currTargets->m_gaussians[k].m_mean;
        X.col(1) = m_currTargets->m_gaussians2[k].m_mean;
        if (tracking_debug) {
          printf("State Interaction -- id: %d\n", m_currTargets->m_gaussians[k].m_track_id);
          printf("Dyna b4: %f;%f;%f;%f cov: %f;%f;%f;%f\n", X(0,0), X(1, 0), X(2, 0), X(3, 0), m_currTargets->m_gaussians[k].m_cov(0,0), m_currTargets->m_gaussians[k].m_cov(1,1), m_currTargets->m_gaussians[k].m_cov(2,2), m_currTargets->m_gaussians[k].m_cov(3,3));
          printf("Sta b4: %f;%f;%f;%f cov: %f; %f;%f;%f\n", X(0,1), X(1, 1), X(2, 1), X(3, 1), m_currTargets->m_gaussians2[k].m_cov(0,0), m_currTargets->m_gaussians2[k].m_cov(1,1), m_currTargets->m_gaussians2[k].m_cov(2,2), m_currTargets->m_gaussians2[k].m_cov(3,3));
        }
        for (size_t j = 0; j < M; j++)
        {
          for (size_t i = 0; i < M; i++)
          {
            C(j) += m_mode_trans(i, j) * Pr(i);
          }
        }
        m_currTargets->m_gaussians[k].c = C(0);
        m_currTargets->m_gaussians2[k].c = C(1);
        if (tracking_debug) {
          printf("D_C: %f, S_C: %f\n", C(0), C(1));
          printf("D_P: %f, S_P: %f\n", Pr(0), Pr(1));
          printf("mode: \n");
          cout << m_mode_trans << endl;
        }
        MatrixXf tmp = X;
        X.fill(0);
        for (size_t j = 0; j < M; j++) {
          for (size_t i = 0; i < M; i++) {
            U(i, j) = 1.0 / C(j) * m_mode_trans(i,j) * Pr(i);
            X.col(j) += tmp.col(i) * U(i, j);    
          }
        }

        for (size_t j = 0; j < M; j++) {
          Eigen::MatrixXf P = Eigen::MatrixXf::Zero(S, S);
          if (j == 0)
          {
            Eigen::VectorXf s0 = tmp.col(0) - X.col(j);
            Eigen::VectorXf s1 = tmp.col(1) - X.col(j);
            if (tracking_debug)
              printf("Mu: %f d-d error: %f %f %f %f; Mu: %f s-d error: %f %f %f %f\n", U(0,j), s0(0), s0(1), s0(2), s0(3), U(1,j), s1(0), s1(1), s1(2), s1(3));
            P += U(0, j) * (m_currTargets->m_gaussians[k].m_cov + s0 * s0.transpose());
            P += U(1, j) * (m_currTargets->m_gaussians2[k].m_cov + s1* s1.transpose());
            m_currTargets->m_gaussians[k].m_cov = P;
            m_currTargets->m_gaussians[k].m_mean = X.col(j);
          }
          else
          {
            Eigen::VectorXf s0 = tmp.col(0) - X.col(j);
            Eigen::VectorXf s1 = tmp.col(1) - X.col(j);
            if (tracking_debug)
              printf("Mu: %f d-s error: %f %f %f %f; Mu: %f s-s error: %f %f %f %f\n", U(0,j), s0(0), s0(1), s0(2), s0(3), U(1,j), s1(0), s1(1), s1(2), s1(3));
            P += U(0, j) * (m_currTargets->m_gaussians[k].m_cov + s0 * s0.transpose());
            P += U(1, j) * (m_currTargets->m_gaussians2[k].m_cov + s1* s1.transpose());
            m_currTargets->m_gaussians2[k].m_cov = P;
            m_currTargets->m_gaussians2[k].m_mean = X.col(j);
          }
        }

        // for (size_t i = 0; i < M; i++) {
        //   Eigen::MatrixXf P = Eigen::MatrixXf::Zero(S, S);
        //   if (i == 0)
        //   {
        //     for (size_t j = 0; j < M; j++) {
        //       Eigen::VectorXf s = tmp.col(i) - X.col(j);
        //       printf("Mu: %f Dynamic error: %f %f %f %f\n", U(i,j), s(0), s(1), s(2), s(3));
        //       P += U(i, j) * (m_currTargets->m_gaussians[k].m_cov + s * s.transpose());
        //     }
        //     m_currTargets->m_gaussians[k].m_cov = P;
        //     m_currTargets->m_gaussians[k].m_mean = X.col(i);
        //   }
        //   else
        //   {
        //     for (size_t j = 0; j < M; j++) {
        //       Eigen::VectorXf s = tmp.col(i) - X.col(j);
        //       printf("Mu: %f Static error: %f %f %f %f\n", U(i,j), s(0), s(1), s(2), s(3));
        //       P += U(i, j) * (m_currTargets->m_gaussians2[k].m_cov + s * s.transpose());
        //     }
        //     m_currTargets->m_gaussians2[k].m_cov = P;
        //     m_currTargets->m_gaussians2[k].m_mean = X.col(i);
        //   }
        // }
        if (tracking_debug) {
          printf("Dyna after: %f;%f;%f;%f cov: %f;%f;%f;%f\n", X(0,0), X(1, 0), X(2, 0), X(3, 0), m_currTargets->m_gaussians[k].m_cov(0,0), m_currTargets->m_gaussians[k].m_cov(1,1), m_currTargets->m_gaussians[k].m_cov(2,2), m_currTargets->m_gaussians[k].m_cov(3,3));
          printf("Sta after: %f;%f;%f;%f cov: %f; %f;%f;%f\n", X(0,1), X(1, 1), X(2, 1), X(3, 1), m_currTargets->m_gaussians2[k].m_cov(0,0), m_currTargets->m_gaussians2[k].m_cov(1,1), m_currTargets->m_gaussians2[k].m_cov(2,2), m_currTargets->m_gaussians2[k].m_cov(3,3));
        }
      }
    }

    void updateModelProb()
    {
      for (size_t i = 0; i < m_currTargets->m_gaussians.size(); ++i)
      {
        float c_sum = 0;
        c_sum += m_currTargets->m_gaussians[i].m_weight * m_currTargets->m_gaussians[i].c;
        c_sum += m_currTargets->m_gaussians2[i].m_weight * m_currTargets->m_gaussians2[i].c;
        m_currTargets->m_gaussians[i].model_prob = 1 / c_sum * m_currTargets->m_gaussians[i].m_weight * m_currTargets->m_gaussians[i].c;
        m_currTargets->m_gaussians2[i].model_prob = 1 / c_sum * m_currTargets->m_gaussians2[i].m_weight * m_currTargets->m_gaussians2[i].c;
        if (tracking_debug) {
          printf("---id: %d\n", m_currTargets->m_gaussians[i].m_track_id);
          printf("Update model %d prob: %f %f %f\n", m_currTargets->m_gaussians[i].model_type, m_currTargets->m_gaussians[i].m_weight, m_currTargets->m_gaussians[i].c, m_currTargets->m_gaussians[i].model_prob);
          printf("Update model %d prob: %f %f %f\n", m_currTargets->m_gaussians2[i].model_type, m_currTargets->m_gaussians2[i].m_weight, m_currTargets->m_gaussians2[i].c, m_currTargets->m_gaussians2[i].model_prob);
        }
      }
    }

    void estimateFusion() 
    {
      m_currTargets->m_fusedGaussians.clear();

      for (size_t i = 0; i < m_currTargets->m_gaussians.size(); ++i)
      {
        GaussianModel<S> fused;
        fused.m_track_id = m_currTargets->m_gaussians[i].m_track_id;
        // compare weights of two model
        fused.m_weight = m_currTargets->m_gaussians[i].m_weight * m_currTargets->m_gaussians[i].model_prob +
                          m_currTargets->m_gaussians2[i].m_weight * m_currTargets->m_gaussians2[i].model_prob;
        if (m_currTargets->m_gaussians[i].model_prob > m_currTargets->m_gaussians2[i].model_prob)
        {
          fused.model_type = m_currTargets->m_gaussians[i].model_type;
          fused.model_prob = m_currTargets->m_gaussians[i].model_prob;
        }
        else
        {
          fused.model_type = m_currTargets->m_gaussians2[i].model_type;
          fused.model_prob = m_currTargets->m_gaussians2[i].model_prob;
        }
        for (size_t j = 0; j < S; ++j)
        {
          fused.m_mean[j] = m_currTargets->m_gaussians[i].m_mean[j] * m_currTargets->m_gaussians[i].model_prob + 
                            m_currTargets->m_gaussians2[i].m_mean[j] * m_currTargets->m_gaussians2[i].model_prob;
        }
        if (tracking_debug) {
          printf("Fusion -- id: %d, type: %d, prob: %f\n", fused.m_track_id, fused.model_type, fused.model_prob);
          printf("Dyna: %f;%f;%f;%f  Sta: %f;%f;%f;%f\n", m_currTargets->m_gaussians[i].m_mean[0], m_currTargets->m_gaussians[i].m_mean[1], m_currTargets->m_gaussians[i].m_mean[2], m_currTargets->m_gaussians[i].m_mean[3],
          m_currTargets->m_gaussians2[i].m_mean[0], m_currTargets->m_gaussians2[i].m_mean[1], m_currTargets->m_gaussians2[i].m_mean[2], m_currTargets->m_gaussians2[i].m_mean[3]);
        }
        m_currTargets->m_fusedGaussians.emplace_back(fused);
      }
      assert(m_currTargets->m_gaussians.size() == m_currTargets->m_fusedGaussians.size());
      assert(m_currTargets->m_gaussians2.size() == m_currTargets->m_fusedGaussians.size());
    }

    void propagate()
    {
      m_nPredTargets = 0;
      m_nPropagated = 0;
      m_nSpawned = 0;

      // IMM State interaction 
      stateInteraction();

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
      
      if (tracking_debug) {
        printf("After after normalized\n");
        for (size_t i = 0; i < m_currTargets->m_gaussians.size(); ++i){
          printf("Meas idx: %d\n", i / m_expTargets->m_gaussians.size());
          printf("id: %d weights: %f\n", m_currTargets->m_gaussians[i].m_track_id, m_currTargets->m_gaussians[i].m_weight);
        }
      }

      pruneGaussians(ordered_indices);

      updateModelProb();

      // IMM State Fusion
      estimateFusion();

      // Clean std::vectors :
      m_expMeasure.clear();
      m_expDisp.clear();
      m_uncertainty.clear();
      m_covariance.clear();

      m_expMeasure2.clear();
      m_expDisp2.clear();
      m_uncertainty2.clear();
      m_covariance2.clear();
      // Show trackers
      // showTrackerList();
    }

    void reset()
    {
      m_currTargets->m_gaussians.clear();
      m_currTargets->m_gaussians2.clear();
      m_currTargets->m_fusedGaussians.clear();
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

    void pruneGaussians(const std::vector<size_t> indices)
    {
      associated.clear();
      GaussianMixture<S> pruned_targets;
      pruned_targets.m_gaussians.clear(); // dynamic
      pruned_targets.m_gaussians2.clear(); // static
      std::vector<bool> merge_checker(indices.size(), false);
      std::vector<int> i_close_to_best;
      std::vector<int> i_close_to_corres;
      int n_meas = m_measTargets->m_gaussians.size();
      int n_tracker = m_expTargets->m_gaussians.size();
      int merge_cnt = 0;

      for (size_t i = 0; i < indices.size(); ++i)
      {
        size_t i_best = indices[i];
        if (i_best == -1 || m_currTargets->m_gaussians[i_best].m_weight < m_pruneTruncThld)
          break;

        int i_corres = getCorrespondingIndex(i_best);
        int t_best = m_currTargets->m_gaussians[i_best].model_type;
        int t_corres = m_currTargets->m_gaussians[i_corres].model_type;
        int track_id = m_currTargets->m_gaussians[i_best].m_track_id;
        float c_best = m_currTargets->m_gaussians[i_best].c;
        float c_corres = m_currTargets->m_gaussians[i_corres].c;
        if (tracking_debug) {
          printf("---id: %d, best: %d, corres: %d\n", m_currTargets->m_gaussians[i_best].m_track_id, t_best, t_corres);
          printf("b_idx: %d, b_weight: %f, c_idx: %d, c_weight: %f\n", i_best, m_currTargets->m_gaussians[i_best].m_weight, i_corres, m_currTargets->m_gaussians[i_corres].m_weight);
          printf("x: %f; y: %f; vx: %f; vy: %f\n", m_currTargets->m_gaussians[i_best].m_mean[0], m_currTargets->m_gaussians[i_best].m_mean[1], m_currTargets->m_gaussians[i_best].m_mean[2], m_currTargets->m_gaussians[i_best].m_mean[3]);
        }
        if (merge_checker[i_best]) {
          if (tracking_debug)
            printf("%d already merged\n", i_best);
          continue;
        }

        GaussianModel<S> best_model;
        GaussianModel<S> corres_model;
        i_close_to_best.clear();
        i_close_to_corres.clear();

        // Birth gaussian
        if (track_id < 0) 
        {
          if (tracking_debug)
            printf("Birth tracker %d: -1 -> %d\n", t_best, m_ntrack); // t_best = -1
          assert(t_best == -1);
          m_currTargets->m_gaussians[i_best].m_track_id = m_ntrack++; 
          track_id = m_currTargets->m_gaussians[i_best].m_track_id;
          best_model = m_currTargets->m_gaussians[i_best];
          corres_model = m_currTargets->m_gaussians[i_corres];
          corres_model.m_track_id = track_id;
          assert(corres_model.m_track_id == best_model.m_track_id);
          best_model.c = c_best;
          corres_model.c = c_corres;
          // pruned_targets.m_gaussians.emplace_back(std::move(best_model));
          // pruned_targets.m_gaussians2.emplace_back(std::move(corres_model));
        }
        // Normal tracker
        else 
        {
          // Find closest gaussian
          TicToc closest_time;
          std::vector<GaussianModel<S>> target_gaussians = m_currTargets->m_gaussians;
          float gauss_distance;
          Matrix<float, D, 1> point_vec;
          Matrix<float, D, 1> mean_vec;
          Matrix<float, D, D> cov;
          i_close_to_best.push_back(i_best);
          i_close_to_corres.push_back(i_corres);
          
          merge_checker[i_best] = true;
          // merge_cnt++;

          for (size_t j = 0; j < target_gaussians.size(); ++j)
          {
            if (j != i_best && !merge_checker[j])
            {
              // check model type
              if (target_gaussians[j].model_type != t_best)
                continue;
              else
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
                  if (tracking_debug)
                    printf("compared id: %d; %f; %f %f %f; dist: %f\n", target_gaussians[j].m_track_id, target_gaussians[j].m_weight, target_gaussians[j].m_mean[0], target_gaussians[j].m_mean[1], 0.0, gauss_distance);
                  i_close_to_best.push_back(j);
                  i_close_to_corres.push_back(getCorrespondingIndex(j));
                }
              }
            }
          }
          if (tracking_debug)
            printf("curr tracker: %d; %f %f %f %f; best type: %d\n", track_id, target_gaussians[i_best].m_mean[0], target_gaussians[i_best].m_mean[1], target_gaussians[i_best].m_mean[2], target_gaussians[i_best].m_mean[3], target_gaussians[i_best].model_type);
          // ROS_WARN("Find closest: %f ms\n", closest_time.toc());

          // Merge close gaussians
          TicToc merge_time;

          best_model.clear();
          corres_model.clear();

          if (i_close_to_best.size() > 1 && i_close_to_corres.size() > 1)
          {
            for (size_t j = 0; j < i_close_to_best.size(); j++)
            {
              best_model.m_weight += target_gaussians[i_close_to_best[j]].m_weight;
              corres_model.m_weight += target_gaussians[i_close_to_corres[j]].m_weight;
            }

            for (size_t j = 0; j < i_close_to_best.size(); j++)
            {
              best_model.m_mean += target_gaussians[i_close_to_best[j]].m_mean * target_gaussians[i_close_to_best[j]].m_weight;
              corres_model.m_mean += target_gaussians[i_close_to_corres[j]].m_mean * target_gaussians[i_close_to_corres[j]].m_weight;
            }

            if (best_model.m_weight != 0.f)
              best_model.m_mean /= best_model.m_weight;
            if (corres_model.m_weight != 0.f)
              corres_model.m_mean /= corres_model.m_weight;

            best_model.m_cov.setZero();
            corres_model.m_cov.setZero();

            for (size_t j = 0; j < i_close_to_best.size(); j++)
            {
              Matrix<float, S, 1> diff = best_model.m_mean - target_gaussians[i_close_to_best[j]].m_mean;
              best_model.m_cov += target_gaussians[i_close_to_best[j]].m_weight * (target_gaussians[i_close_to_best[j]].m_cov + diff * diff.transpose());
              diff = corres_model.m_mean - target_gaussians[i_close_to_corres[j]].m_mean;
              corres_model.m_cov += target_gaussians[i_close_to_corres[j]].m_weight * (target_gaussians[i_close_to_corres[j]].m_cov + diff * diff.transpose());
            }

            if (best_model.m_weight != 0.f)
              best_model.m_cov /= best_model.m_weight;
            if (corres_model.m_weight != 0.f)
              corres_model.m_cov /= corres_model.m_weight;
          } else {
            best_model = target_gaussians[i_close_to_best[0]];
            corres_model = target_gaussians[i_close_to_corres[0]];
          }

          best_model.model_type = t_best;
          corres_model.model_type = t_corres;
          best_model.m_track_id = track_id;
          corres_model.m_track_id = track_id;
          best_model.c = c_best;
          corres_model.c = c_corres;

          assert(i_close_to_corres.size() == i_close_to_best.size());
          assert(target_gaussians[i_best].m_track_id == target_gaussians[i_corres].m_track_id);
          assert(i_close_to_best[0] == i_best);
          assert(i_close_to_corres[0] == i_corres);
          assert(best_model.model_type == t_best);
          assert(corres_model.model_type == t_corres);
          assert(best_model.c != 0.0);
          assert(corres_model.c != 0.0);

          // ROS_WARN("Insert merged time: %f ms", merge_time.toc());
        }

        // add survived tracker = propagated + birth + not propagated
        if (i_best < n_tracker)
        {
          if (tracking_debug) {
            printf("Survived track id: %d\n", track_id);
            printf("type: %d, track weight: %f, %f %f %f %f\n", best_model.model_type, best_model.m_weight, best_model.m_mean[0], best_model.m_mean[1], best_model.m_cov(0,0), best_model.m_cov(1,1));
            printf("type: %d, track weight: %f, %f %f %f %f\n", corres_model.model_type, corres_model.m_weight, corres_model.m_mean[0], corres_model.m_mean[1], corres_model.m_cov(0,0), corres_model.m_cov(1,1));
          }
          if (associated.find(track_id) == associated.end())
          {
            associated.insert(std::make_pair(track_id, -1));
            addTracker(best_model);
            if (t_best == 0) // static
            {
              if (tracking_debug)
                printf("Best is static\n");
              pruned_targets.m_gaussians.emplace_back(corres_model);
              pruned_targets.m_gaussians2.emplace_back(best_model);
              assert(best_model.model_type == 0);
              assert(corres_model.model_type == 1);
              assert(pruned_targets.m_gaussians.back().model_type == 1);
              assert(pruned_targets.m_gaussians2.back().model_type == 0);
            }
            else if (t_best == 1) // dynamic
            {
              if (tracking_debug)
                printf("Best is dynamic\n");
              pruned_targets.m_gaussians.emplace_back(best_model);
              pruned_targets.m_gaussians2.emplace_back(corres_model);
              assert(best_model.model_type == 1);
              assert(corres_model.model_type == 0);
              assert(pruned_targets.m_gaussians.back().model_type == 1);
              assert(pruned_targets.m_gaussians2.back().model_type == 0);
            }
            else // birth
            {
              if (tracking_debug)
                printf("Best is birth\n");
              assert(best_model.model_type == -1);
              assert(corres_model.model_type == -1);
              pruned_targets.m_gaussians.emplace_back(best_model);
              pruned_targets.m_gaussians2.emplace_back(corres_model);
            }
          } 
          else
          {
            if (tracking_debug)
              printf("Already associated\n");
          }
        }
        else // add measurement updated tracker
        {
          size_t i_meas = (i_best - n_tracker) / n_tracker;
          int meas_id = m_measTargets->m_gaussians[i_meas].m_track_id;
          if (tracking_debug) {
            printf("New track_id: %d, meas_id: %d\n", track_id, meas_id);
            printf("type: %d, id: %d, track weight: %f, %f %f %f %f\n", best_model.model_type, best_model.m_track_id, best_model.m_weight, best_model.m_mean[0], best_model.m_mean[1], best_model.m_cov(0,0), best_model.m_cov(1,1));
            printf("type: %d, id: %d, track weight: %f, %f %f %f %f\n", corres_model.model_type, corres_model.m_track_id, corres_model.m_weight, corres_model.m_mean[0], corres_model.m_mean[1], corres_model.m_cov(0,0), corres_model.m_cov(1,1));
          }
          if (associated.find(track_id) == associated.end()) // new tracker
          {
            addTracker(best_model);
            associated.insert(std::make_pair(track_id, meas_id));
            if (t_best == 0) // static
            {
              if (tracking_debug)
                printf("Best is static\n");
              pruned_targets.m_gaussians.emplace_back(corres_model);
              pruned_targets.m_gaussians2.emplace_back(best_model);
              assert(best_model.model_type == 0);
              assert(corres_model.model_type == 1);
              assert(pruned_targets.m_gaussians.back().model_type == 1);
              assert(pruned_targets.m_gaussians2.back().model_type == 0);
            }
            else if (t_best == 1) // dynamic
            {
              if (tracking_debug)
                printf("Best is dynamic\n");
              pruned_targets.m_gaussians.emplace_back(best_model);
              pruned_targets.m_gaussians2.emplace_back(corres_model);
              assert(best_model.model_type == 1);
              assert(corres_model.model_type == 0);
              assert(pruned_targets.m_gaussians.back().model_type == 1);
              assert(pruned_targets.m_gaussians2.back().model_type == 0);
            }
            else // birth
            {
              if (tracking_debug)
                printf("Best is birth\n");
              assert(best_model.model_type == -1);
              assert(corres_model.model_type == -1);
              pruned_targets.m_gaussians.emplace_back(best_model);
              pruned_targets.m_gaussians2.emplace_back(corres_model);
            }
          }
          else
          {
            if (tracking_debug)
              printf("Already associated\n");       
          }          
        }
          // printf("Last added best model: %d\n", pruned_targets.m_gaussians.back().model_type);

        if (pruned_targets.m_gaussians.size() > m_nMaxPrune)
          break;
      }
      ROS_WARN("%d out of %d has merged", merge_cnt, (int)indices.size());


      // For measurements that are not associated, add new tracker
      size_t birth_idx = m_nPredTargets - m_spawnTargets->m_gaussians.size() - m_birthTargets->m_gaussians.size();
      for (size_t k = 0; k < m_measTargets->m_gaussians.size(); ++k)
      {
        int meas_id = m_measTargets->m_gaussians[k].m_track_id;
        auto result = std::find_if(associated.begin(), associated.end(), 
          [meas_id](const auto &element){return element.second == meas_id;});
        if (result == associated.end()) // not matched
        {
          size_t gauss_idx = m_nPredTargets * (k+1) + birth_idx;
          // printf("Birth idx: %d, Gauss_idx: %d, exp size: %d\n", birth_idx, gauss_idx, m_expTargets->m_gaussians.size());
          GaussianModel<S> birth_gaussian = m_currTargets->m_gaussians[gauss_idx];

          birth_gaussian.m_track_id = m_ntrack++;
          birth_gaussian.m_weight = birth_gaussian.m_weight < 0.2f ? 0.2f: 0.95 * birth_gaussian.m_weight;
          assert(birth_gaussian.model_type == -1);
          assert(birth_gaussian.c != 0.0);
          assert(birth_gaussian.m_track_id >= 0);
          pruned_targets.m_gaussians.emplace_back(birth_gaussian);
          pruned_targets.m_gaussians2.emplace_back(birth_gaussian);
          if (tracking_debug) {
              printf("Meas id: %d, %f;%f;0.0   birth gaussian: %d, %f; %f; 0.0   weight: %f\n", 
            meas_id, m_measTargets->m_gaussians[k].m_mean[0], m_measTargets->m_gaussians[k].m_mean[1],
            birth_gaussian.m_track_id, birth_gaussian.m_mean[0], birth_gaussian.m_mean[1], birth_gaussian.m_weight);
          }
        }
      }

      // add pruned gaussians
      m_currTargets->m_gaussians = pruned_targets.m_gaussians;
      m_currTargets->m_gaussians2 = pruned_targets.m_gaussians2;
      if (tracking_debug)
        printf("Current target size: %d\n", (int)m_currTargets->m_gaussians.size());
      frame_count++;
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
      if (tracking_debug)
        printf("Number of predicted targets: %d\n", m_nPredTargets);
      // m_expMeasure.clear();
      // m_expMeasure.reserve(m_nPredTargets);

      // m_expDisp.clear();
      // m_expDisp.reserve(m_nPredTargets);

      // m_uncertainty.clear();
      // m_uncertainty.reserve(m_nPredTargets);

      // m_covariance.clear();
      // m_covariance.reserve(m_nPredTargets);

      m_expMeasure2.clear();
      m_expMeasure2.reserve(m_nPredTargets);

      m_expDisp2.clear();
      m_expDisp2.reserve(m_nPredTargets);

      m_uncertainty2.clear();
      m_uncertainty2.reserve(m_nPredTargets);

      m_covariance2.clear();
      m_covariance2.reserve(m_nPredTargets);

      // printf("Observation cov: %f %f %f %f\n", m_obsCov(0,0), m_obsCov(1,1), m_obsCov(2,2), m_obsCov(3,3));
      // for (auto const &tgt : m_expTargets->m_gaussians)
      // {
      //   // Compute the expected measurement
      //   m_expMeasure.push_back(m_obsMat * tgt.m_mean);
      //   m_expDisp.push_back(m_obsCov + m_obsMat * tgt.m_cov * m_obsMatT);
      //   m_uncertainty.push_back(tgt.m_cov * m_obsMatT * m_expDisp.back().inverse());
      //   m_covariance.push_back((Matrix<float, S, S>::Identity() - m_uncertainty.back() * m_obsMat) * tgt.m_cov);
      //   printf("Expected: -- id: %d, tgt cov: %f;%f;%f;%f\n", tgt.m_track_id, tgt.m_cov(0,0), tgt.m_cov(1,1), tgt.m_cov(2,2), tgt.m_cov(3,3));
      //   printf("expMeas: %f %f %f %f\n",m_expMeasure.back()[0], m_expMeasure.back()[1], m_expMeasure.back()[2], m_expMeasure.back()[3]);
      //   printf("expDisp: %f %f %f %f\n",m_expDisp.back()(0,0), m_expDisp.back()(1,1), m_expDisp.back()(2,2), m_expDisp.back()(3,3));
      //   printf("m_cov: %f %f %f %f\n", m_covariance.back()(0,0), m_covariance.back()(1,1), m_covariance.back()(2,2), m_covariance.back()(3,3));
      //   // printf("Expected: %f;%f; vel: %f; %f; id: %d\n", m_expMeasure.back()[0], m_expMeasure.back()[1], m_expMeasure.back()[2], m_expMeasure.back()[3], tgt.m_track_id);
      // }

      // observation2
      if (tracking_debug) 
        printf("Observation cov: %f %f\n", m_obsCov2(0,0), m_obsCov2(1,1));
      for (auto const &tgt : m_expTargets->m_gaussians)
      {
        // Compute the expected measurement
        m_expMeasure2.push_back(m_obsMat2 * tgt.m_mean); // <D, 1> : H*xp
        m_expDisp2.push_back(m_obsCov2 + m_obsMat2 * tgt.m_cov * m_obsMat2T); // <D, D> : R + H*Pp*H'
        m_uncertainty2.push_back(tgt.m_cov * m_obsMat2T * m_expDisp2.back().inverse()); // <S, D> : K = P*H'*inv(expDisp)
        m_covariance2.push_back((Matrix<float, S, S>::Identity() - m_uncertainty2.back() * m_obsMat2) * tgt.m_cov); // <S, D>
        Matrix<float, S, D> PpH = tgt.m_cov * m_obsMat2T;
        if (tracking_debug) {
          printf("Expected: -- id: %d; %f %f %f %f\n", tgt.m_track_id, tgt.m_mean[0], tgt.m_mean[1], tgt.m_mean[2], tgt.m_mean[3]);
          cout << "Pp: " << endl << tgt.m_cov << endl;
          cout << "Hx: " << endl << m_expMeasure2.back() << endl;
          cout << "R+H*Pp*H':" << endl << m_expDisp2.back() << endl;
          cout << "PpH'" << endl << PpH << endl;
          cout << "K: " << endl << m_uncertainty2.back() << endl;
          cout << "P: " << endl << m_covariance2.back() << endl;
        }
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
      if (tracking_debug)
        printf("Spawn: \n");
      for (auto const &curr : m_currTargets->m_gaussians2) // static object
      {
        for (auto const &spawn : m_spawnModels)
        {
          GaussianModel<S> new_spawn;

          // Define a gaussian model from the existing target
          // and spawning properties
          // new_spawn.m_weight = curr.m_weight * spawn.m_weight;
          new_spawn.m_weight = m_pSurvival * curr.m_weight;
          new_spawn.m_mean = spawn.m_offset + spawn.m_trans * curr.m_mean;
          new_spawn.m_cov = spawn.m_cov + spawn.m_trans * curr.m_cov * spawn.m_trans.transpose();
          // new_spawn.m_isFalseTarget = true;
          new_spawn.m_track_id = curr.m_track_id;
          new_spawn.model_prob = curr.model_prob;
          new_spawn.model_type = 0; // static
          new_spawn.c = curr.c;
          if (tracking_debug)
            printf("id: %d; %f %f %f %f; %f %f %f %f\n", new_spawn.m_track_id, 
          curr.m_cov(0, 0), curr.m_cov(1,1), curr.m_cov(2,2), curr.m_cov(3,3),
          new_spawn.m_cov(0,0), new_spawn.m_cov(1,1), new_spawn.m_cov(2,2), new_spawn.m_cov(3,3));
          // Add this new gaussian to the list of expected targets
          m_spawnTargets->m_gaussians.push_back(new_spawn);
          ++m_nSpawned;
          // Update the number of expected targets
          ++m_nPredTargets;
        }
      }
      if (tracking_debug)
        printf("\n");
    }

    void predictTargets()
    {
      m_expTargets->m_gaussians.clear();
      m_expTargets->m_gaussians.reserve(m_currTargets->m_gaussians.size());
      if (tracking_debug)
        printf("Targets: \n");
      for (auto const &curr : m_currTargets->m_gaussians) // dynamic objects
      {
        // Compute the new shape of the target
        GaussianModel<S> new_target;
        new_target.m_weight = m_pSurvival * curr.m_weight;
        new_target.m_mean = m_tgtDynTrans * curr.m_mean;
        new_target.m_cov = m_tgtDynCov + m_tgtDynTrans * curr.m_cov * m_tgtDynTrans.transpose();
        new_target.m_track_id = curr.m_track_id;
        new_target.model_prob = curr.model_prob;
        new_target.model_type = 1; // constant velocity
        new_target.c = curr.c;
        if (tracking_debug)
          printf("id: %d, %f %f %f %f\n", new_target.m_track_id, new_target.m_cov(0,0), new_target.m_cov(1,1), new_target.m_cov(2,2), new_target.m_cov(3,3));
        // Push back to the expected targets
        m_expTargets->m_gaussians.push_back(new_target);
        ++m_nPredTargets;
        ++m_nPropagated;
      }
      if (tracking_debug)
        printf("\n");
    }

    int getCorrespondingIndex(int idx)
    {
      int idx_in_match = idx % (int)m_expTargets->m_gaussians.size();
      
      if (idx_in_match < m_nPropagated) // spawned
      {
        return idx + m_nPropagated + 1;
      }
      else if (idx_in_match > m_nPropagated) // propagated
      {
        return idx - m_nPropagated - 1;
      }
      else
      {
        return idx;
      }
    }

    void update()
    {
      m_currTargets->m_gaussians.clear();
      m_currTargets->m_gaussians2.clear();
      // We'll consider every possible association : std::vector size is (expected targets)*(measured targets)
      m_currTargets->m_gaussians.reserve((m_measTargets->m_gaussians.size() + 1) *
                                         m_expTargets->m_gaussians.size());
      // m_currTargets->m_gaussians.reserve((m_measTargets->m_gaussians.size()) *
      //                                    m_expTargets->m_gaussians.size());

      if (tracking_debug)
        printf("Update - meas %d, exp: %d\n", (int)m_measTargets->m_gaussians.size(), (int)m_expTargets->m_gaussians.size());

      // First set of gaussians : mere propagation of existing ones
      // don't propagate the "birth" targets... we set their weight to 0

      for (auto const &target : m_expTargets->m_gaussians)
      {
        // Copy the target into the final set, adjust the weight if it was spawned
        auto newTarget = target;
        // newTarget.m_weight = target.m_isFalseTarget ? 0.f : (1.f - m_pDetection) * target.m_weight;
        newTarget.m_weight = target.m_isFalseTarget ? 0.f : 0.3 * target.m_weight;
        newTarget.m_track_id = target.m_track_id;
        newTarget.model_prob = target.model_prob;
        newTarget.model_type = target.model_type;
        newTarget.c = target.c;
        m_currTargets->m_gaussians.emplace_back(std::move(newTarget));
      }

      uint cur_id = 0;
      uint meas_cnt = 0;
      std::vector<int> best_indices;
      // Second set of gaussians : match observations and previsions
      for (auto &measuredTarget : m_measTargets->m_gaussians)
      {
        uint start_normalize = m_currTargets->m_gaussians.size();
        if (tracking_debug)
          printf("Measurement association: %d\n", meas_cnt);
        meas_cnt++;
        for (uint n_targt = 0; n_targt < m_expTargets->m_gaussians.size(); ++n_targt)
        {

          // Compute matching factor between predictions and measures (2)
          const auto distance = mahalanobis<2>(measuredTarget.m_mean.template head<D>(),
                                               m_expMeasure2[n_targt],
                                               m_expDisp2[n_targt]);
          GaussianModel<S> matchTarget;

          matchTarget.m_weight = m_pDetection * m_expTargets->m_gaussians[n_targt].m_weight / distance;

          matchTarget.m_mean = m_expTargets->m_gaussians[n_targt].m_mean +
                               m_uncertainty2[n_targt] * (measuredTarget.m_mean.template head<D>() - m_expMeasure2[n_targt]);

          matchTarget.m_cov = m_covariance2[n_targt];
          matchTarget.m_track_id = m_expTargets->m_gaussians[n_targt].m_track_id;
          // matchTarget.model_prob = 00; // TO-DO
          matchTarget.model_type = m_expTargets->m_gaussians[n_targt].model_type;
          matchTarget.c = m_expTargets->m_gaussians[n_targt].c;
          if (tracking_debug) {
            printf("matched -- id: %d, distance: %f, weight: %f\n", matchTarget.m_track_id, distance, matchTarget.m_weight);
            printf("target mean: %f;%f;%f;%f\n", m_expTargets->m_gaussians[n_targt].m_mean[0], m_expTargets->m_gaussians[n_targt].m_mean[1], m_expTargets->m_gaussians[n_targt].m_mean[2],m_expTargets->m_gaussians[n_targt].m_mean[3]);
            printf("target cov: %f %f %f %f\n", m_expTargets->m_gaussians[n_targt].m_cov(0,0), m_expTargets->m_gaussians[n_targt].m_cov(1, 1), m_expTargets->m_gaussians[n_targt].m_cov(2,2), m_expTargets->m_gaussians[n_targt].m_cov(3,3));
            printf("meas mean: %f;%f;%f;%f;\n", measuredTarget.m_mean[0], measuredTarget.m_mean[1], measuredTarget.m_mean[2], measuredTarget.m_mean[3]);
            printf("updated mean: %f;%f;%f;%f\n", matchTarget.m_mean[0], matchTarget.m_mean[1], matchTarget.m_mean[2], matchTarget.m_mean[3]);
            printf("updated cov: %f %f %f %f\n", matchTarget.m_cov(0,0),matchTarget.m_cov(1,1), matchTarget.m_cov(2,2), matchTarget.m_cov(3,3));
            // printf("distance: %f, weight: %f, mean: %f;%f\n", distance, m_expTargets->m_gaussians[n_targt].m_weight, matchTarget.m_mean[0], matchTarget.m_mean[1]);
          }
          m_currTargets->m_gaussians.emplace_back(std::move(matchTarget));

          // // Compute matching factor between predictions and measures.
          // const auto distance = mahalanobis<2>(measuredTarget.m_mean.template head<D>(),
          //                                      m_expMeasure[n_targt].template head<D>(),
          //                                      m_expDisp[n_targt].template topLeftCorner<D, D>());
          // GaussianModel<S> matchTarget;

          // matchTarget.m_weight = m_pDetection * m_expTargets->m_gaussians[n_targt].m_weight / distance;

          // matchTarget.m_mean = m_expTargets->m_gaussians[n_targt].m_mean +
          //                      m_uncertainty[n_targt] * (measuredTarget.m_mean - m_expMeasure[n_targt]);

          // matchTarget.m_cov = m_covariance[n_targt];
          // matchTarget.m_track_id = m_expTargets->m_gaussians[n_targt].m_track_id;
          // // matchTarget.model_prob = 00; // TO-DO
          // matchTarget.model_type = m_expTargets->m_gaussians[n_targt].model_type;
          // matchTarget.c = m_expTargets->m_gaussians[n_targt].c;
          // printf("matched -- id: %d, distance: %f, weight: %f\n", matchTarget.m_track_id, distance, matchTarget.m_weight);
          // printf("exp target: %f;%f;%f;%f\n", m_expTargets->m_gaussians[n_targt].m_mean[0], m_expTargets->m_gaussians[n_targt].m_mean[1], m_expTargets->m_gaussians[n_targt].m_mean[2],m_expTargets->m_gaussians[n_targt].m_mean[3]);
          // printf("meas mean: %f;%f;%f;%f, expMeasure: %f;%f;%f;%f ", measuredTarget.m_mean[0], measuredTarget.m_mean[1], measuredTarget.m_mean[2], measuredTarget.m_mean[3], m_expMeasure[n_targt][0], m_expMeasure[n_targt][1], m_expMeasure[n_targt][2], m_expMeasure[n_targt][3]);
          // printf("expDisp: %f; %f;\n", m_expDisp[n_targt](0,0), m_expDisp[n_targt](1,1));
          // // printf("diff: %f; %f; %f; %f\n", measuredTarget.m_mean[0] - m_expMeasure[n_targt][0], measuredTarget.m_mean[1]-m_expMeasure[n_targt][1], measuredTarget.m_mean[2]-m_expMeasure[n_targt][2], measuredTarget.m_mean[3]-m_expMeasure[n_targt][3]);
          // // std::cout << "cov: " << std::endl;
          // // std::cout << m_uncertainty[n_targt] <<std::endl;
          // printf("matched mean: %f;%f;%f;%f\n", matchTarget.m_mean[0], matchTarget.m_mean[1], matchTarget.m_mean[2], matchTarget.m_mean[3]);
          // // printf("distance: %f, weight: %f, mean: %f;%f\n", distance, m_expTargets->m_gaussians[n_targt].m_weight, matchTarget.m_mean[0], matchTarget.m_mean[1]);

          // m_currTargets->m_gaussians.emplace_back(std::move(matchTarget));
        }
        cur_id++;
        // Normalize weights in the same predicted set, taking clutter into account
        m_currTargets->normalize(m_measNoiseBackground, start_normalize,
                                 m_currTargets->m_gaussians.size(), 1);
        // printf("After normalized\n");
        
        // for (size_t t = start_normalize; t < m_currTargets->m_gaussians.size(); t++)
        // {
        //   printf("---id: %d, weight: %f, %f;%f;%f;%f\n", m_currTargets->m_gaussians[t].m_track_id, m_currTargets->m_gaussians[t].m_weight, m_currTargets->m_gaussians[t].m_mean[0], m_currTargets->m_gaussians[t].m_mean[1], m_currTargets->m_gaussians[t].m_mean[2], m_currTargets->m_gaussians[t].m_mean[3]);
        // }
      }
      if (tracking_debug) {
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
    Matrix<float, S, S> m_tgtStaTrans;
    Matrix<float, S, S> m_tgtStaCov;

    Matrix<float, S, S> m_dynTrans;
    Matrix<float, S, S> m_dynCov;
    Matrix<float, S, S> m_staTrans;
    Matrix<float, S, S> m_staCov;

    Matrix<float, S, S> m_obsMat;
    Matrix<float, S, S> m_obsMatT;
    Matrix<float, S, S> m_obsCov;

    Matrix<float, D, S> m_obsMat2;
    Matrix<float, S, D> m_obsMat2T;
    Matrix<float, D, D> m_obsCov2; 

    // Temporary matrices, used for the update process
    std::vector<Matrix<float, S, S>> m_covariance;
    std::vector<Matrix<float, S, 1>> m_expMeasure;
    std::vector<Matrix<float, S, S>> m_expDisp;
    std::vector<Matrix<float, S, S>> m_uncertainty;

    std::vector<Matrix<float, S, S>> m_covariance2;
    std::vector<Matrix<float, D, 1>> m_expMeasure2;
    std::vector<Matrix<float, D, D>> m_expDisp2;
    std::vector<Matrix<float, S, D>> m_uncertainty2;

    std::unique_ptr<GaussianMixture<S>> m_birthModel;

    std::unique_ptr<GaussianMixture<S>> m_birthTargets;
    std::unique_ptr<GaussianMixture<S>> m_currTargets;
    std::unique_ptr<GaussianMixture<S>> m_expTargets;
    std::unique_ptr<GaussianMixture<S>> m_extractedTargets;
    std::unique_ptr<GaussianMixture<S>> m_measTargets;
    std::unique_ptr<GaussianMixture<S>> m_spawnTargets;
    std::unique_ptr<GaussianMixture<S>> m_tempTargets;

    int m_nPropagated;
    int m_nSpawned;
    int frame_count;
    list<TrackerPerId<D>> trackerList;
    Matrix<float, M, M> m_mode_trans;
    std::map<int, int> associated;
  };
}