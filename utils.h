//
// Created by arvie on 17/04/25.
//

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <random>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <boost/functional/hash.hpp> // for tuple hashing function
#include <tuple>

using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

// Timing
chrono::time_point<high_resolution_clock> time_before() {
    return high_resolution_clock::now();
}

void time_after(chrono::time_point<high_resolution_clock> t1, const string &msg = "") {
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << msg << ": " << ms_double.count() << "ms, " << ms_double.count() * 0.001 << "s" << std::endl;
}


class SampleIntPartition{
    /*
     * For k-hops, integer partition where some size Q is partitioned into exactly N parts.
     * We then can uniformly sample from these partitions to construct k-hops.
     */
public:
    unordered_map<tuple<int, int>, int, boost::hash<std::tuple<int, int>>> QN_cache;
    unordered_map<tuple<int, int, int>, int, boost::hash<std::tuple<int, int, int>>> QNK_cache;

    int suggested_cache_size;
    int max_cache_size;

    SampleIntPartition(const int suggested_cache_size=10000000, const int max_cache_size=10){

        this->suggested_cache_size = suggested_cache_size;
        this->max_cache_size = suggested_cache_size * max_cache_size;
        // QN_cache.reserve(this->suggested_cache_size);
        // QNK_cache.reserve(this->suggested_cache_size);
    }

    void print(){
        string s = "";
        s += "QN cache size: " + to_string(QN_cache.size()) + ",\t";
        s += "QNK cache size: " + to_string(QNK_cache.size());
        cout << s << endl;
    }

    void clear_if_needed(){
        if (QN_cache.size() > suggested_cache_size){
            QN_cache.clear();
        }
        if (QNK_cache.size() > suggested_cache_size){
            QNK_cache.clear();
        }
    }

    int partition_QN(const int Q, const int N){
        auto key = make_tuple(Q, N);
        if (QN_cache.find(key) != QN_cache.end()){
            return QN_cache[key];
        }
        if (N == 0){
            return Q == 0 ? 1 : 0;
        }
        if (N > Q || Q <= 0){
            return 0;
        }
        if (N == Q){
            return 1;
        }
        int result = partition_QN(Q - 1, N - 1) + partition_QN(Q - N, N);
        if (QN_cache.size() > max_cache_size){
            QN_cache.clear();
        }
        QN_cache[key] = result;
        return result;
    };

    int partition_QNK(const int Q, const int N, const int K){
        auto key = make_tuple(Q, N, K);
        if (QNK_cache.find(key) != QNK_cache.end()){
            return QNK_cache[key];
        }
        if (Q > N * K || K <= 0){
            return 0;
        }
        if (Q == N * K){
            return 1;
        }
        int result = 0;
        for (int i = 0; i < N; i++){
            result += partition_QNK(Q - i * K, N - i, K - 1);
        }
        if (QNK_cache.size() > max_cache_size){
            QNK_cache.clear();
        }
        QNK_cache[key] = result;
        return result;
    };

    int min_max(const int n, const int s){
        auto min_max = static_cast<int>(floor(n / s));
        if (n % s > 0){
            min_max += 1;
        }
        return min_max;
    }

    void uniform_random_partition(int Q, int N, vector<int> &segment_lengths, std::mt19937 &gen, const bool shuffle=true){
        /*
        file:///h/arvie/Downloads/Partitioning_paper1.pdf
        @article{loceyrandom,
            title={Random integer partitions with restricted numbers of parts},
            author={Locey, Kenneth J}
        }
         */
        clear_if_needed();
        segment_lengths.clear();
        int _min = min_max(Q, N);
        int _max = Q - N + 1;
        int total = partition_QN(Q, N);
        std::uniform_int_distribution<int> dist(1, total);
        int which = dist(gen);
        while (Q > 0){
            int K_chosen = _min;
            int count = 0;
            for (int K = _min; K <= _max; K++){
                count = partition_QNK(Q, N, K);
                if (count >= which){
                    count = partition_QNK(Q, N, K - 1);
                    K_chosen = K;
                    break;
                }
            }
            segment_lengths.push_back(K_chosen);
            Q -= K_chosen;
            if (Q <= 0){  // change from == for safety
                break;
            }
            N -= 1;
            which -= count;
            _min = min_max(Q, N);
            _max = K_chosen;
        }
        if (shuffle){
            std::shuffle(segment_lengths.begin(), segment_lengths.end(), gen);
        }
    }

    void non_uniform_random_partition(int Q, int N, vector<int> &segment_lengths, std::mt19937 &gen, const bool shuffle=true) {
        /*
         * A faster non-uniform version
         */
        clear_if_needed();
        segment_lengths = vector<int>(N, 1);
        int remaining_length = Q - N;
        for (int i = 0; i < N - 1; i++) {
            if (remaining_length <= 0) {
                break;
            }
            std::uniform_int_distribution<int> dist(0, remaining_length);
            int r = dist(gen);
            segment_lengths[i] += r;
            remaining_length -= r;
        }
        segment_lengths[-1] += remaining_length;
        if (shuffle){
            std::shuffle(segment_lengths.begin(), segment_lengths.end(), gen);
        } else {
            std::sort(segment_lengths.begin(), segment_lengths.end());
        }
    }

};

#endif //UTILS_H
