/*
	DBSCAN Algorithm
	15S103182
	Ethan
*/
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <stack>
#include <assert.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <numeric>
// #define NDEBUG

using namespace std;
using namespace cv;

const int NOISE=-3;
const int NOT_CLASSIFIED=-1;
double xmin=__DBL_MAX__;
double ymin=__DBL_MAX__;
double xmax=__DBL_MIN__;
double ymax=__DBL_MIN__;
int w = 400, h = 400;

struct Pnt{
    double x,y;
    int nearPtsCnt, cluster;
    double getDis(const Pnt& other){
        return sqrt(pow(other.x-x, 2)+pow(other.y-y, 2));
    }
};

class DBSCAN{
    private:
        int minPts, size, clusterIdx;
        double eps;
        vector<Pnt> points;
        vector<vector<int> > nearPntsIdx, cluster;
    public:
        DBSCAN(double eps_, int minPts_, vector<Pnt> points_):
            eps(eps_), minPts(minPts_), points(points_){
                clusterIdx=-1;
                size=points.size();
                nearPntsIdx.resize(size);
        }

        void run(){
            std::cout<<"run dbscan\n";
            checkNearPnts();
            for (int i = 0; i < size; i++)
            {
                if(points[i].cluster!= NOT_CLASSIFIED)continue;
                if(isCorePnt(i)){
                    dfs(i, ++clusterIdx);
                }
                else {
                    points[i].cluster=NOISE;
                } 
            }
            cluster.resize(clusterIdx+1);
            for(int i=0;i<size;i++) {
                if(points[i].cluster != NOISE) {
                    cluster[points[i].cluster].push_back(i);
                }
            }
            std::cout<<"finish run dbscan\n";
        }

        void dfs(int idx, int clusterIdx){
            points[idx].cluster = clusterIdx;
            if(!isCorePnt(idx))return;

            for(auto& next:nearPntsIdx[idx]){
                if(points[next].cluster==NOT_CLASSIFIED)
                    dfs(next, clusterIdx);
            }
        }

        bool isCorePnt(int idx){return points[idx].nearPtsCnt >= minPts;}

        void checkNearPnts(){
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    if (i==j)continue;
                    if(points[i].getDis(points[j])<eps)
                    {
                        points[i].nearPtsCnt++;
                        nearPntsIdx[i].push_back(j);
                    }
                    
                }
                
            }
        }

        vector<vector<Pnt> > getcluster(){
            vector<vector<Pnt> > ans;
            int cluster_size = cluster.size();
            ans.resize(cluster_size);
            for (int i = 0; i < cluster_size; i++)
            {
                for (auto& c:cluster[i])
                {
                    ans[i].push_back(points[c]);
                }
                
            }
            std::cout<<"cluster_size:"<<cluster_size<<"\n";
            return ans;
        }
};

class InputReader{
    private:
        ifstream in;
        vector<Pnt> points;
    public:
        InputReader(string inputFile){
            in.open(inputFile);
            assert(in);
            int idx;
            double x,y;
            std::cout<<"read file\n";
            while (!in.eof())
            {
                in >> idx >> x >> y;
                xmax = max(x, xmax);
                xmin = min(x, xmin);
                ymax = max(y, ymax);
                ymin = min(y, ymin);
                points.push_back({x, y, 0, NOT_CLASSIFIED});
                // std::cout<<"["<<x<<","<<y<<"]\n";
            }
        }
        vector<Pnt> getPnts(){
            return points;
        }
};

int getx(double x_in){
    return w*(x_in-xmin)/(xmax-xmin);
}

int gety(double y_in){
    return w*(y_in-ymin)/(ymax-ymin);
}

void show(vector<vector<Pnt>> clusters){
    int cluster_size = clusters.size();
    vector<cv::Scalar> scalar(cluster_size);
    for (int i = 0; i < cluster_size; i++)
    {
        int n=(255*3*i/cluster_size);
        int c=std::max(0, (n-255*2)%255);
        int b=std::max(0, (n-255*1)%255);
        int a=std::max(0, (n-255*0)%255);
        scalar[i] = cv::Scalar(a, b, c);
    }
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(255, 255, 255));

    for(int i = 0; i < cluster_size; i++){
        for(auto& p:clusters[i]){
            cv::circle(img, cv::Point(getx(p.x), gety(p.y)), 2, scalar[i], -1);
            // std::cout<<"["<<getx(p.x)<<","<<gety(p.y)<<"]\n";
        }
    }
    cv::imshow("hwb", img);
    cv::waitKey(1000000);
}



vector<double> FitLine(vector<double> x, vector<double> y)
{
    assert(x.size()==y.size());
    int n=x.size();
    double mX=std::accumulate(x.begin(), x.end(), 0)/(double)n;
    double mY=std::accumulate(y.begin(), y.end(), 0)/(double)n;
    
    double a = 0.0f;
    double b = 0.0f;
    double c = 0.0f;

    double sXX = 0.0f;
    double sXY = 0.0f;
    double sYY = 0.0f;

    for (size_t i = 0; i < n; i++)
    {
        sXX += ((double)x[i] - mX) * ((double)x[i] - mX);
        sXY += ((double)x[i] - mX) * ((double)y[i] - mY);
        sYY += ((double)y[i] - mY) * ((double)y[i] - mY);
    }
    
    bool isVertical = sXY == 0 && sXX < sYY;
    bool isHorizontal = sXY == 0 && sXX > sYY;
    bool isIndeterminate = sXY == 0 && sXX == sYY;
    double slope = nan("1");
    double intercept = nan("1");

    if (isVertical)
    {
        a = 1.0f;
        b = 0.0f;
        c = (float)mX;
    }
    else if (isHorizontal)
    {
        a = 0.0f;
        b = 1.0f;
        c = (float)mY;
    }
    else if (isIndeterminate)
    {
        a = nan("1");
        b = nan("1");
        c = nan("1");
    }
    else
    {
        slope = (sYY - sXX + sqrt((sYY - sXX) * (sYY - sXX) + 4.0 * sXY * sXY)) / (2.0 * sXY);  //斜率
        intercept = mY - slope * mX;                                                            //截距
        double normFactor = (intercept >= 0.0 ? 1.0 : -1.0) * sqrt(slope * slope + 1.0);
        a = (float)(-slope / normFactor);
        b = (float)(1.0 / normFactor);
        c = (float)(intercept / normFactor);
    }
    std::cout<<"a:"<<a<<" b:"<<b<<" c:"<<c<<" nan:"<<nan("1")<<" isnan:"<<isnan(nan("1"))<<"\n";
    return vector<double>{a,b,c};
}

double getDisP2Line(vector<double> param, double x, double y){
    assert(param.size()==3);
    double a=param[0], b=param[1], c=param[2];
    return (a*x + b*y + c)/sqrt(a*a + b*b);
}

int main(int argc, const char* argv[]){
    assert(argc==5);

    string inputFile(argv[1]), n(argv[2]), eps(argv[3]), minPts(argv[4]);
    InputReader in(inputFile);

    DBSCAN dbscan(stod(eps), stoi(minPts), in.getPnts());
    dbscan.run();

    show(dbscan.getcluster());
    // OutputPrinter out();    
    // FitLine({1,2}, {1,2});
    // std::cout<<"dis:"<<getDisP2Line({1,1,1}, 0, 0)<<"\n";
}
