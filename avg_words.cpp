#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <omp.h>

#include <filesystem>
#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <sstream>
#include <algorithm>
#include <limits>
#include <climits>
#include <atomic>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;

using pii = std::pair<int, std::string>;

int countWords(const std::string &text)
{
    std::stringstream ss(text);
    std::string word;
    int cnt = 0;
    while (ss >> word)
        cnt++;
    return cnt;
}

struct ThreadData
{
    long long totalWords = 0;
    int maxWords = 0;
    int minWords = INT_MAX;
    bool initFailed = false;

    std::priority_queue<pii, std::vector<pii>, std::greater<pii>> largest20;
    std::priority_queue<pii> smallest20;
};

// Helper function to render a tqdm-style progress bar
void printProgressBar(int current, int total, std::chrono::steady_clock::time_point startTime)
{
    static constexpr int barWidth = 30;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);

    auto now = std::chrono::steady_clock::now();
    double elapsedSecs = std::chrono::duration<double>(now - startTime).count();
    double rate = current / (elapsedSecs > 0 ? elapsedSecs : 1.0);

    #pragma omp critical(ProgressBarLock)
    {
        std::cout << "\rProcessing: [";
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "% "
                  << current << "/" << total << " ["
                  << std::fixed << std::setprecision(1) << rate << " it/s]" 
                  << std::flush;

        if (current == total)
            std::cout << "\n";
    }
}

int main()
{
    std::vector<std::string> images;

    for (const auto &entry : fs::directory_iterator("./News"))
    {
        images.push_back(entry.path().string());
    }

    int n = images.size();

    if (n == 0)
    {
        std::cout << "No images found.\n";
        return 0;
    }

    int numThreads = omp_get_max_threads();
    std::vector<ThreadData> threadData(numThreads);

    // Atomic counter for thread-safe progress tracking
    std::atomic<int> completedTasks{0};
    auto startTime = std::chrono::steady_clock::now();

    // Initial print
    printProgressBar(0, n, startTime);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ThreadData &local = threadData[tid];

        tesseract::TessBaseAPI api;

        if (api.Init(nullptr, "eng", tesseract::OEM_LSTM_ONLY))
        {
            #pragma omp critical
            {
                std::cerr << "\nFailed to initialize Tesseract on thread " << tid << "\n";
            }
            local.initFailed = true;
        }
        else
        {
            api.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);

#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < n; i++)
            {
                Pix *image = pixRead(images[i].c_str());

                int words = 0;
                if (image)
                {
                    api.SetImage(image);
                    char *text = api.GetUTF8Text();

                    words = countWords(text ? text : "");

                    local.totalWords += words;
                    local.maxWords = std::max(local.maxWords, words);
                    local.minWords = std::min(local.minWords, words);

                    local.largest20.push({words, images[i]});
                    if (local.largest20.size() > 20)
                        local.largest20.pop();

                    local.smallest20.push({words, images[i]});
                    if (local.smallest20.size() > 20)
                        local.smallest20.pop();

                    delete[] text;
                    pixDestroy(&image);
                }

                // Increment progress atomically and update display
                int currentCompleted = ++completedTasks;
                printProgressBar(currentCompleted, n, startTime);
            }

            api.End();
        }
    }

    // Abort if any thread failed initialization
    for (const auto &t : threadData)
    {
        if (t.initFailed)
        {
            std::cerr << "One or more threads failed to initialize Tesseract. Aborting.\n";
            return 1;
        }
    }

    //---------------------------------------------------
    // Merge thread-local results
    //---------------------------------------------------

    long long totalWords = 0;
    int maxWords = 0;
    int minWords = INT_MAX;

    std::priority_queue<pii, std::vector<pii>, std::greater<pii>> globalLargest20;
    std::priority_queue<pii> globalSmallest20;

    for (auto &t : threadData)
    {
        totalWords += t.totalWords;
        maxWords = std::max(maxWords, t.maxWords);
        minWords = std::min(minWords, t.minWords);

        while (!t.largest20.empty())
        {
            globalLargest20.push(t.largest20.top());
            if (globalLargest20.size() > 20)
                globalLargest20.pop();
            t.largest20.pop();
        }

        while (!t.smallest20.empty())
        {
            globalSmallest20.push(t.smallest20.top());
            if (globalSmallest20.size() > 20)
                globalSmallest20.pop();
            t.smallest20.pop();
        }
    }

    //---------------------------------------------------
    // Print statistics
    //---------------------------------------------------

    std::cout << "\nImages processed : " << n << '\n';
    std::cout << "Average words    : " << static_cast<double>(totalWords) / n << '\n';
    std::cout << "Maximum words    : " << maxWords << '\n';
    std::cout << "Minimum words    : " << (minWords == INT_MAX ? 0 : minWords) << '\n';

    //---------------------------------------------------
    // Extract heaps into vectors for sorted printing
    //---------------------------------------------------

    std::vector<pii> largest;
    std::vector<pii> smallest;

    while (!globalLargest20.empty())
    {
        largest.push_back(globalLargest20.top());
        globalLargest20.pop();
    }

    while (!globalSmallest20.empty())
    {
        smallest.push_back(globalSmallest20.top());
        globalSmallest20.pop();
    }

    std::sort(largest.begin(), largest.end(), [](const pii &a, const pii &b) {
        return a.first > b.first;
    });

    std::sort(smallest.begin(), smallest.end(), [](const pii &a, const pii &b) {
        return a.first < b.first;
    });

    std::cout << "\nTop 20 Largest\n--------------------------\n";
    for (const auto &[count, file] : largest)
    {
        std::cout << count << "  " << file << '\n';
    }

    std::cout << "\nTop 20 Smallest\n--------------------------\n";
    for (const auto &[count, file] : smallest)
    {
        std::cout << count << "  " << file << '\n';
    }

    return 0;
}