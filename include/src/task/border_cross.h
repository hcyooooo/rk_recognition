// File: border_cross.h
// Created Date: 2021-01-13 15:00:00
#ifndef BORDER_CROSS_H
#define BORDER_CROSS_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>
#include "types.h"
// // 定义点结构
// struct Point {
//     int x;
//     int y;
// };

// 定义多边形为点的集合
typedef std::vector<Point> Polygon;

// 函数声明
bool onSegment(Point p, Point q, Point r);
int orientation(Point p, Point q, Point r);
bool doIntersect(Point p1, Point q1, Point p2, Point q2);
bool isInside(Polygon polygon, Point p);

// 新增：带记忆的内部检测函数
// 需要提供一个唯一ID来跟踪对象，以及当前坐标点
bool isInsideWithMemory(Polygon polygon, Point p, int objectId);

// 新增：重置所有跟踪状态
void resetInsideTracking();

#endif