#include "common.h"
#include <cmath>
#include <cstring>

static std::vector<StObject> m_objs;

// const double DIST_THRES = 0.00001;
const double DIST_THRES = 5;


double CalPtDist(StObject& obj1, StObject& obj2)
{
    double dx = obj2.longitude - obj1.longitude;
    double dy = obj2.latitude - obj1.latitude;

    double dist = std::sqrt(dx * dx + dy * dy);
    printf("dist:%f\n", dist);

    return dist;
}

void UpdateObjPos(StObject& obj1, StObject& obj2)
{
    obj1.longitude = (obj1.longitude + obj2.longitude)/2;
    obj1.latitude = (obj1.latitude + obj2.latitude)/2;
}

void fusion(std::vector<StObject>& objs)
{
    std::vector<int> matched(100, 0);
    std::vector<StObject> UnmatchedObjs;
    std::vector<int> m_unMatchedDetObjIdx(50,1);
    // memset(m_unMatchedDetObjIdx, 1, sizeof(m_unMatchedDetObjIdx));

    // for(auto& obj:objs)
    for(int j=0;j<objs.size();j++)
    {
        for(int i=0;i<m_objs.size();i++)
        {
            if(CalPtDist(m_objs[i], objs[j]) < DIST_THRES && m_objs[i].clsId == objs[j].clsId && matched[j] == 0)
            {
                UpdateObjPos(m_objs[i], objs[j]);
                matched[i] = 1;
                m_unMatchedDetObjIdx[j] = 0;
                break;
            }
            // else
            // {
            //     UnmatchedObjs.push_back(obj);
            // }
        }
    }

    //process unmatched
    for(int i=0;i<objs.size();i++)
    {
        printf("m_unMatchedDetObjIdx[%d]=:%d\n",i, m_unMatchedDetObjIdx[i]);
        if(m_unMatchedDetObjIdx[i] == 1)
        {
            m_objs.push_back(objs[i]);
        }
    }

    // m_objs.insert(m_objs.end(), UnmatchedObjs.begin(), UnmatchedObjs.end());

    printf("m_objs size:%d\n", m_objs.size());

}

void GetFusRet(std::vector<StObject> &ret)
{
    
}
