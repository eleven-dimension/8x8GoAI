#include "GameField.h"

std::string act_to_str(int act)
{
	if (act == 0) return "pass";
	std::string str(2, '0');
	str[0] = (act - 1) % WIDTH + 'A';
	str[1] = WIDTH - (act - 1) / WIDTH + '0';
	return str;
}

int str_to_act(std::string str)
{
	if (str == "pass" || str == "PASS") return PASS;
	return 1 + str[0] - 'A' + WIDTH * (WIDTH - str[1] + '0');
}

bool str_valid(std::string str)
{
	if (str == "pass" || str == "PASS") return true;
	if (str.size() != 2 || str[0] < 'A' || str[0] > 'A' + WIDTH || str[1] < '1' || str[1] > '0' + WIDTH) return false;
	return true;
}

std::pair<int, int> act_to_xy(int act)
{
	if (act == PASS) return { -1, -1 };
	return { (act - 1) / WIDTH, (act - 1) % WIDTH };
}

int xy_to_act(int x, int y)
{
	if (x == -1 && y == -1) return PASS;
	return x * WIDTH + y + 1;
}

bool out_field(int pos, int index)
{
	std::pair<int, int> xy = act_to_xy(pos);
	int nx = xy.first + dx[index], ny = xy.second + dy[index];
	if (nx < 0 || nx >= WIDTH || ny < 0 || ny >= WIDTH)
	{
		return true;
	}
	return false;
}

std::vector<double> get_noise(int num)
{
	std::mt19937 g;
	double k = 1;
	double theta = 1;
	double sm = 0;
	auto res = std::vector<double>(num, 0);
	for (int i = 0; i < num; i++)
	{
		std::gamma_distribution<> d(k, theta);
		res[i] = d(g);
		sm += res[i];
	}
	for (int i = 0; i < num; i++) res[i] /= sm;
	return res;
}

GameField::GameField()
{
	this->gameField = empty_field;
	this->pass_cnt = 0;
	this->current_color = black;
}

GameField::GameField(const GameField& field)
{
	this->gameField = field.gameField;
	this->pass_cnt = field.pass_cnt;
	this->current_color = field.current_color;
	this->past_situation_map = field.past_situation_map;
}

void GameField::clear()
{
	this->gameField = empty_field;
}

void GameField::destroy()
{
	this->gameField = empty_field;
	this->pass_cnt = 0;
	this->current_color = black;
	this->past_situation_map = std::map<std::pair<board_type, int>, bool>();
}

void GameField::print()
{
	std::cerr << "---------game field----------" << std::endl;
	for (int i = 0; i < WIDTH; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			int pos = xy_to_act(i, j);
			if (gameField[pos] == blank) std::cerr << '.';
			else if (gameField[pos] == black) std::cerr << 'X';
			else std::cerr << 'O';
		}
		std::cerr << std::endl;
	}
	std::cerr << "------------------------------" << std::endl;
}

std::vector<int> GameField::slice_group(int pos)
{
	std::vector<bool> visit(MAX, false);
	int color = gameField[pos];

	std::queue<int> q;
	std::vector<int> group;
	visit[pos] = true;
	q.push(pos);
	group.emplace_back(pos);

	while (q.size())
	{
		int here = q.front(); q.pop();
		for (int d = 0; d < directions_cnt; d++)
		{
			if (out_field(here, d)) continue;
			int tmp = here + directions[d];
			if (visit[tmp] || gameField[tmp] != color) continue;
			visit[tmp] = true;
			q.push(tmp);
			group.emplace_back(tmp);
		}
	}
	return group;
}

int GameField::count_liberty(std::vector<int> group)
{
	std::vector<bool> visit(MAX, false);
	int cnt = 0;
	for (auto l : group)
	{
		for (int d = 0; d < directions_cnt; d++)
		{
			if (out_field(l, d)) continue;
			int tmp = l + directions[d];
			if (visit[tmp]) continue;
			if (gameField[tmp] == blank)
			{
				visit[tmp] = true;
				cnt++;
			}
		}
	}
	return cnt;
}

void GameField::take(std::vector<int> group)
{
	for (auto l : group) gameField[l] = blank;
}

std::pair<bool, GameField::board_type> GameField::settle(int act, int color, bool just_try)
{
	int opposer = -color;
	if (act == PASS) return { true, gameField };

	int pos = act;
	if (gameField[pos] != blank) return { false, gameField };

	GameField nxt_field(*this);
	nxt_field.gameField[pos] = color;

	for (int d = 0; d < directions_cnt; d++)
	{
		if (out_field(pos, d)) continue;
		int tmp = pos + directions[d];
		if (nxt_field.gameField[tmp] == opposer)
		{
			std::vector<int> affects = nxt_field.slice_group(tmp);
			int liberty = nxt_field.count_liberty(affects);
			if (liberty < 1) nxt_field.take(affects);
		}
	}

	std::vector<int> my_group = nxt_field.slice_group(pos);

	int liberty = nxt_field.count_liberty(my_group);

	if (liberty < 1) return { false, gameField };
	else if (just_try) {

		return { true, gameField };
	}
	else return { true, nxt_field.gameField };
}

void GameField::fill_blank(int pos)
{
	std::queue<int> q;
	q.push(pos);
	gameField[pos] = mark;
	while (q.size())
	{
		int here = q.front(); q.pop();
		for (int d = 0; d < directions_cnt; d++)
		{
			if (out_field(here, d)) continue;
			int tmp = here + directions[d];
			if (gameField[tmp] != blank) continue;
			gameField[tmp] = mark;
			q.push(tmp);
		}
	}
}

std::pair<bool, bool> GameField::decide_blank_whose()
{
	bool near_black = false, near_white = false;
	for (int l = 1; l < ALL; l++)
	{
		if (near_black && near_white) break;
		if (gameField[l] == mark)
		{
			for (int d = 0; d < directions_cnt; d++)
			{
				if (out_field(l, d)) continue;
				int tmp = l + directions[d];
				if (!near_black && gameField[tmp] == black)
					near_black = true;
				if (!near_white && gameField[tmp] == white)
					near_white = true;
			}
		}
	}
	return { near_black, near_white };
}

double GameField::count_stones()
{
	GameField copy_field(*this);
	while (true)
	{
		int fst_blank = -1;
		for (int i = 1; i < ALL; i++)
		{
			if (copy_field.gameField[i] == blank)
			{
				fst_blank = i;
				break;
			}
		}
		if (fst_blank == -1) break;

		copy_field.fill_blank(fst_blank);
		auto near_info = copy_field.decide_blank_whose();
		bool near_black = near_info.first;
		bool near_white = near_info.second;

		if (near_black && !near_white)
		{
			for (auto& l : copy_field.gameField)
			{
				if (l == mark) l = black;
			}
		}
		else if (!near_black && near_white)
		{
			for (auto& l : copy_field.gameField)
			{
				if (l == mark) l = white;
			}
		}
		else
		{
			for (auto& l : copy_field.gameField)
			{
				if (l == mark) l = draw;
			}
		}
	}
	int black_territory = 0, draw_territory = 0;
	for (auto i : copy_field.gameField)
	{
		if (i == black) black_territory++;
		else if (i == draw)	draw_territory++;
	}
	return black_territory + draw_territory / 2.0;
}

int GameField::referee()
{
	if (pass_cnt >= 2)
	{
		auto black_score = count_stones();
		if (black_score > THRESHOLD) return black_win;
		return white_win;
	}
	return unfinished;
}


void GameField::play(int act, int color)
{
	if (act == PASS)
	{
		pass_cnt++;
		past_situation_map[std::make_pair(gameField, -color)] = true;
	}
	else
	{
		pass_cnt = 0;
		gameField = settle(act, color).second;
		past_situation_map[std::make_pair(gameField, -color)] = true;
	}

	this->current_color = -color;
}

void GameField::play(std::string act_str, int color)
{
	int act = str_to_act(act_str);
	play(act, color);
}

void GameField::play(int act)
{
	if (act == PASS)
	{
		pass_cnt++;
		past_situation_map[{gameField, -this->current_color}] = true;
	}
	else
	{
		pass_cnt = 0;
		gameField = settle(act, this->current_color).second;
		past_situation_map[{gameField, -this->current_color}] = true;
	}
	this->current_color = -this->current_color;
}

void GameField::play(std::string act_str)
{
	int act = str_to_act(act_str);
	play(act);
}

std::vector<int> GameField::valid_moves(int color)
{
	std::vector<int> valid_moves = { PASS };
	for (int pos = 1; pos < ALL; pos++)
	{
		auto move_info = settle(pos, color);
		auto good_move = move_info.first;
		auto nxt_field = move_info.second;
		if (good_move && past_situation_map[{nxt_field, -color}] == false)
			valid_moves.emplace_back(pos);
	}
	return valid_moves;
}

std::vector<int> GameField::valid_moves_mask(int color)
{
	std::vector<int> mask(ALL, 0);
	mask[PASS] = 1;
	for (int pos = 1; pos < ALL; pos++)
	{
		auto move_info = settle(pos, color);
		auto good_move = move_info.first;
		auto nxt_field = move_info.second;
		if (good_move && past_situation_map[{nxt_field, -color}] == false)
			mask[pos] = 1;
	}
	return mask;
}

std::vector<double> GameField::get_gamefield_mat()
{
	std::vector<double> input_vector(3 * WIDTH * WIDTH, 0);

	auto get_linear_index = [](int a, int b, int c) { return a * WIDTH * WIDTH + b * WIDTH + c; };

	if (current_color == white)
		for (int x = 0; x < WIDTH; x++)
			for (int y = 0; y < WIDTH; y++)
				input_vector[get_linear_index(2, x, y)] = 1;

	for (int i = 1; i < gameField.size(); i++)
	{		
		auto pos_pair = act_to_xy(i);
		int x = pos_pair.first, y = pos_pair.second;
		if (gameField[i] == black)
		{
			input_vector[get_linear_index(0, x, y)] = 1;
		}
		else if (gameField[i] == white)
		{
			input_vector[get_linear_index(1, x, y)] = 1;
		}
	}
	return input_vector;
}
