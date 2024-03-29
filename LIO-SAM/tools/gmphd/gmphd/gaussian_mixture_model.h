#pragma once
// Author : Benjamin Lefaudeux (blefaudeux@github)

#include "eigen_tools.h"
#include <list>
#include <algorithm>
#include <numeric>

namespace gmphd
{
    using namespace std;
    using namespace Eigen;

    template <size_t D>
    struct GaussianModel
    {
        GaussianModel()
        {
            clear();
        }

        GaussianModel &operator=(const GaussianModel &rhs)
        {
            if (this != &rhs)
            {
                m_mean = rhs.m_mean;
                m_cov = rhs.m_cov;
                m_weight = rhs.m_weight;
                m_track_id = rhs.m_track_id;
                m_isFalseTarget = rhs.m_isFalseTarget;
                model_prob = rhs.model_prob;
                model_type = rhs.model_type;
                c = rhs.c;
            }

            return *this;
        }

        void clear()
        {
            m_mean.setZero();
            m_cov.setIdentity();
            m_weight = 0.f;
            m_track_id = -1;
            model_prob = 0.5;
            model_type = -1;
            m_isFalseTarget = false;
            c = 0.5;
        }

        float m_weight;
        Matrix<float, D, 1> m_mean;
        Matrix<float, D, D> m_cov;
        bool m_isFalseTarget;
        int m_track_id;
        float model_prob;
        int model_type;
        float c;
    };

    /*!
    * \brief The gaussian_mixture is a sum of gaussian models,
    *  with according weights. Everything is public, no need to get/set...
    */
    template <size_t D>
    class GaussianMixture
    {
    public:
        GaussianMixture()
        {
            m_gaussians.clear();
            m_gaussians2.clear();
            m_fusedGaussians.clear();
        }

        GaussianMixture(GaussianMixture const &source)
        {
            m_gaussians = source.m_gaussians;
            m_gaussians2 = source.m_gaussians2;
            m_fusedGaussians = source.m_fusedGaussians;
        }

        GaussianMixture(vector<GaussianModel<D>> const &source)
        {
            m_gaussians = source;
        }

        GaussianMixture operator=(const GaussianMixture &source)
        {
            // Skip assignment if same object
            if (this == &source)
                return *this;

            // Else, use vectors & Eigen "=" operator
            m_gaussians = source.m_gaussians;
            m_gaussians2 = source.m_gaussians2;
            m_fusedGaussians = source.m_fusedGaussians;
            return *this;
        }

        GaussianModel<D> mergeGaussians(vector<int> &i_gaussians_to_merge, bool b_remove_from_mixture)
        {
            // TODO: Ben - rewrite this crap, could be half way long

            GaussianModel<D> merged_model;
            Matrix<float, D, 1> diff;

            if (i_gaussians_to_merge.size() > 1)
            {
                // Reset the destination
                merged_model.clear();

                // Build merged gaussian :
                // - weight is the sum of all weights
                for (auto const &i_g : i_gaussians_to_merge)
                {
                    merged_model.m_weight += m_gaussians[i_g].m_weight;
                }

                // - gaussian center is the weighted m_mean of all centers
                for (auto const &i_g : i_gaussians_to_merge)
                {
                    merged_model.m_mean += m_gaussians[i_g].m_mean * m_gaussians[i_g].m_weight;
                }

                if (merged_model.m_weight != 0.f)
                {
                    merged_model.m_mean /= merged_model.m_weight;
                }

                // - covariance is related to initial gaussian model cov and the discrepancy
                // from merged m_mean position and every merged gaussian pose
                merged_model.m_cov.setZero();
                for (auto const &i_g : i_gaussians_to_merge)
                {
                    diff = merged_model.m_mean - m_gaussians[i_g].m_mean;

                    merged_model.m_cov += m_gaussians[i_g].m_weight * (m_gaussians[i_g].m_cov + diff * diff.transpose());
                }

                if (merged_model.m_weight != 0.f)
                {
                    merged_model.m_cov /= merged_model.m_weight;
                }
            }
            else
            {
                // Just return the initial single gaussian model :
                merged_model = m_gaussians[i_gaussians_to_merge[0]];
            }

            if (b_remove_from_mixture)
            {
                // Remove input gaussians from the mixture
                // - sort the index vector
                std::sort(i_gaussians_to_merge.begin(),
                          i_gaussians_to_merge.end());

                // - pop out the corresponding gaussians, in reverse
                auto it = m_gaussians.begin();

                for (int i = i_gaussians_to_merge.size() - 1; i > -1; ++i)
                {
                    m_gaussians.erase(it + i);
                }
            }

            return merged_model;
        }

        void normalize(float linear_offset)
        {
            const float sum = std::accumulate(m_gaussians.begin(), m_gaussians.end(), 0.f, [](const GaussianModel<D> &g1, const GaussianModel<D> &g2) { return g1.m_weight + g2.m_weight; });

            if ((linear_offset + sum) != 0.f)
            {
                for (auto &gaussian : m_gaussians)
                {
                    gaussian.m_weight /= (linear_offset + sum);
                }
            }
        }

        void normalize(float linear_offset, int start_pos, int stop_pos, int step)
        {
            float sum = 0.f;
            for (int i = start_pos; i < stop_pos; ++i)
            {
                sum += m_gaussians[i * step].m_weight;
            }

            if ((linear_offset + sum) != 0.f)
            {
                for (int i = start_pos; i < stop_pos; ++i)
                {
                    m_gaussians[i * step].m_weight /= (linear_offset + sum);
                }
            }
        }

        void prune(float trunc_threshold, float merge_threshold, uint max_gaussians)
        {
            // Sort the gaussians mixture, ascending order
            sort();

            int index, i_best;

            vector<int> i_close_to_best;
            GaussianMixture<D> pruned_targets;
            GaussianModel<D> merged_gaussian;

            merged_gaussian.clear();
            pruned_targets.m_gaussians.clear();

            printf("Before pruning: %d\n", (int)m_gaussians.size());
            std::vector<int> matched_ids;
            while (!m_gaussians.empty() && pruned_targets.m_gaussians.size() < max_gaussians)
            {
                // - Pick the biggest gaussian (based on weight)
                // i_best = selectBestGaussian();
                i_best = selectBestGaussian(matched_ids);
                printf("ibest: %d %f\n", i_best, m_gaussians[i_best].m_weight);
                if (i_best == -1 || m_gaussians[i_best].m_weight < trunc_threshold)
                {
                    printf("no prune\n");
                    break;
                }
                else
                {
                    // - Select all the gaussians close enough, to merge if needed
                    i_close_to_best.clear();
                    selectCloseGaussians(i_best, merge_threshold, i_close_to_best);
                    // - Build a new merged gaussian
                    i_close_to_best.push_back(i_best); // Add the initial gaussian
                    int best_track_id = m_gaussians[i_best].m_track_id;
                    printf("Merge size: %d\n", (int)i_close_to_best.size());
                    if (i_close_to_best.size() > 1)
                    {
                        merged_gaussian = mergeGaussians(i_close_to_best, false);
                    }
                    else
                    {
                        merged_gaussian = m_gaussians[i_close_to_best[0]];
                    }
                    merged_gaussian.m_track_id = best_track_id;
                    
                    printf("Merged: %d\n", merged_gaussian.m_track_id);
                    matched_ids.push_back(merged_gaussian.m_track_id);
                    // - Append merged gaussian to the pruned_targets gaussian mixture
                    pruned_targets.m_gaussians.push_back(merged_gaussian);

                    // - Remove all the merged gaussians from current_targets :
                    // -- Sort the indexes
                    std::sort(i_close_to_best.begin(), i_close_to_best.end());

                    // -- Remove from the last one (to keep previous indexes unchanged)
                    while (!i_close_to_best.empty())
                    {
                        index = i_close_to_best.back();
                        i_close_to_best.pop_back();

                        m_gaussians.erase(m_gaussians.begin() + index);
                    }
                }
            }
            printf("After pruning: %d\n", (int)pruned_targets.m_gaussians.size());

            m_gaussians = pruned_targets.m_gaussians;
        }

        void sort()
        {
            std::sort(m_gaussians.begin(), m_gaussians.end(), [](const auto &lhs, const auto &rhs) {
                return lhs.m_weight > rhs.m_weight;
            });
        }

        void selectCloseGaussians(int i_ref, float threshold, vector<int> &close_gaussians)
        {
            close_gaussians.clear();

            float gauss_distance;

            Matrix<float, D, 1> diff_vec;
            Matrix<float, D, D> cov_inverse;

            // We only take positions into account there
            int i = 0;
            for (auto const &gaussian : m_gaussians)
            {
                if (i != i_ref)
                {
                    // Compute distance
                    diff_vec = m_gaussians[i_ref].m_mean.head(D) -
                               gaussian.m_mean.head(D);

                    cov_inverse = (m_gaussians[i_ref].m_cov.topLeftCorner(D, D)).inverse();

                    gauss_distance = diff_vec.transpose() *
                                     cov_inverse.topLeftCorner(D, D) *
                                     diff_vec;

                    // Add to the set of close gaussians, if below threshold
                    if ((gauss_distance < threshold) && (gaussian.m_weight != 0.f))
                    {
                        close_gaussians.push_back(i);
                    }
                }
                ++i;
            }
        }

        int selectBestGaussian(std::vector<int> matched_ids)
        {
            float best_weight = 0.f;
            int best_index = -1;
            int i = 0;

            std::for_each(m_gaussians.begin(), m_gaussians.end(), [&](GaussianModel<D> const &gaussian) {
                if (gaussian.m_weight > best_weight)
                {
                    auto result = std::find(matched_ids.begin(), matched_ids.end(), gaussian.m_track_id);
                    if (result == matched_ids.end()) // not matched yet
                    {
                        best_weight = gaussian.m_weight;
                        best_index = i;
                    }
                    else
                    {
                        printf("Id: %d already matched, weight: %f\n", gaussian.m_track_id, best_weight);
                    }
                }
                ++i;
            });

            return best_index;  
        }

        int selectBestGaussian()
        {
            float best_weight = 0.f;
            int best_index = -1;
            int i = 0;

            std::for_each(m_gaussians.begin(), m_gaussians.end(), [&](GaussianModel<D> const &gaussian) {
                if (gaussian.m_weight > best_weight)
                {
                    best_weight = gaussian.m_weight;
                    best_index = i;
                }
                ++i;
            });

            return best_index;
        }

        // void changeReferential(const Matrix4f &transform)
        // {
        //     Matrix<float, 4, 1> temp_vec, temp_vec_new;
        //     temp_vec(3, 0) = 1.f;

        //     // Gaussian model :
        //     // - [x, y, z, dx/dt, dy/dt, dz/dt] m_mean values
        //     // - 6x6 covariance

        //     // For every gaussian model, change referential
        //     for (auto &gaussian : m_gaussians)
        //     {
        //         // Change positions
        //         temp_vec.block(0, 0, 3, 1) = gaussian.m_mean.block(0, 0, 3, 1);

        //         temp_vec_new = transform * temp_vec;

        //         gaussian.m_mean.block(0, 0, 3, 1) = temp_vec_new.block(0, 0, 3, 1);

        //         // Change speeds referential
        //         temp_vec.block(0, 0, 3, 1) = gaussian.m_mean.block(3, 0, 3, 1);

        //         temp_vec_new = transform * temp_vec;

        //         gaussian.m_mean.block(3, 0, 3, 1) = temp_vec_new.block(0, 0, 3, 1);

        //         // Change covariance referential
        //         //  (only take the rotation into account)
        //         // TODO
        //     }
        // }

    public:
        vector<GaussianModel<D>> m_gaussians;
        vector<GaussianModel<D>> m_gaussians2;
        vector<GaussianModel<D>> m_fusedGaussians;
    };
} // namespace gmphd