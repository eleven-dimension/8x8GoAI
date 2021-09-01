#pragma once
#include <vector>
#include <string>
#include <map>
#include <queue> 
#include <cmath>
#include <cfloat>
#include <numeric>
#include <iostream>
#include <random>

#include <chrono>
#include <memory>
#include <sstream>
#include <algorithm>
#include <exception>

const double THRESHOLD = 32.75;
const int WIDTH = 8;
const int TOTAL = 64;
const int ALL = 65;
const int MAX = 70;

const int PASS = 0;

const std::vector<int> directions = { -WIDTH, 1, WIDTH, -1 };
const int directions_cnt = 4;
const std::vector<int> directions_index = { 0, 1, 2, 3 };

const int up = 0, right = 1, down = 2, left = 3;

const std::vector<int> dx = { -1, 0, 1, 0 };
const std::vector<int> dy = { 0, 1, 0, -1 };

const int offensive = 0, defensive = 1;

const int blank = -2;
const int black = 1;
const int white = -1;
const int draw = 2;
const int mark = 3;

const int black_win = 1;
const int white_win = -1;
const int unfinished = 0;

const std::vector<int> empty_field(std::vector<int>(ALL, blank));

extern std::string act_to_str(int act);
extern int str_to_act(std::string str);
extern bool str_valid(std::string str);
extern std::pair<int, int> act_to_xy(int act);
extern int xy_to_act(int x, int y);
extern bool out_field(int pos, int index);
extern std::vector<double> get_noise(int num);

class GameField
{
public:
	using board_type = std::vector<int>;

	board_type gameField;
	std::map<std::pair<board_type, int>, bool> past_situation_map;
	int pass_cnt;
	int current_color;

	GameField();
	GameField(const GameField&);

	void clear();
	void destroy();
	void print();
	std::vector<int> slice_group(int pos);
	int count_liberty(std::vector<int> group);
	void take(std::vector<int> group);
	std::pair<bool, board_type> settle(int act, int color, bool just_try = false);
	std::vector<int> valid_moves(int color);
	std::vector<int> valid_moves_mask(int color);
	void fill_blank(int pos);
	std::pair<bool, bool> decide_blank_whose();
	double count_stones();
	int referee();
	std::vector<double> get_gamefield_mat();

	void play(int pos, int color);
	void play(std::string act_str, int color);
	void play(int pos);
	void play(std::string act_str);
};