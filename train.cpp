#include <iostream>
#include <chrono>
#include <fstream>
#include <io.h>
#include <thread>
#include <future>

#include "GameField.h"
#include "MCTS.h"
#include "libtorch.h"

using namespace std;
using namespace chrono;

random_device rd;

const int GAME_PER_TRAIN = 12 * 24;
const int ITERATION_NUM = 1000;
const int CHECKPOINT_FEQ = 60;
const int BATCH_SIZE = 512;

const double SELFPLAY_CPUCT = 5.0;
const int SELFPLAY_RANDOM_TURN = 8;
const int THREAD_NUM = 12;
const double VIRTUAL_LOSS = 3;
const int SELFPLAY_SIMUL_NUM = 800;

const double CONTEST_CPUCT = 3.0;
const int CONTEST_RANDOM_TURN = 12;
const int CONTEST_SIMUL_NUM = 200;
const int CONTEST_GAME_NUM = 2400;

const double NET_PASS_THRESHOLD = 0.55;

const int REMOVE_GAME_CNT = 12 * 8;

void print(vector<double>& v)
{
	for (auto i = 0; i < v.size(); i++)
	{
		cout << act_to_str(i) << " : " << v[i] << endl;
	}
}

string get_best_network()
{
	ifstream fin("./index/best.txt");
	string best_network;
	fin >> best_network;
	return best_network;
}

string get_newest_network()
{
	ifstream fin("./index/newest.txt");
	string newest_network;
	fin >> newest_network;
	return newest_network;
}

string get_random_directory()
{
	string str;
	default_random_engine jws(rd());
	for (int i = 1; i <= 15; i++)
	{
		int offset = jws() % 36;
		str += offset < 10 ? ('0' + offset) : ('A' - 10 + offset);
	}
	return str;
}

template<class T>
void write_file(vector<T> v, string path)
{
	ofstream fout(path.c_str());
	for (auto i : v)
	{
		fout << i << " ";
	}
}

template<class T>
void write_file(vector<vector<T>> v, string path)
{
	ofstream fout(path.c_str());
	for (int i = 0; i < v.size(); i++)
	{
		for (int j = 0; j < v[i].size(); j++)
		{
			fout << v[i][j] << " ";
		}
		fout << endl;
	}
}

template<class T>
void write_file(T x, string path)
{
	ofstream fout(path.c_str(), ofstream::app);
	fout << x << endl;
}

void write_file(string x, string path)
{
	ofstream fout(path.c_str());
	fout << x << endl;
}

int random_choice(vector<int> sample, vector<double> prob)
{
	std::discrete_distribution<size_t> d{ std::begin(prob), std::end(prob) };
	std::default_random_engine rng{ rd() };
	return sample[d(rng)];
}

int best_choice(vector<double> prob)
{
	auto x = std::distance(prob.begin(), max_element(prob.begin(), prob.end()));
	if (x >= ALL)
	{
		cout << "out of range" << endl;
		return 0;
	}
	return x;
}

namespace Rotate
{
	const int RotateNum = 8;
	const double cx = 3.5;
	const double cy = 3.5;
	const vector<double> sin = { 0, 1, 0, -1, 0, 1, 0, -1 };
	const vector<double> cos = { 1, 0, -1, 0, 1, 0, -1, 0 };

	std::pair<int, int> get_new_pos_xy(int x, int y, double ox, double oy, int index)
	{
		if (index >= 4)
		{
			x = WIDTH - x - 1;
		}
		int nx = (x - ox) * cos[index] - (y - oy) * sin[index] + ox;
		int ny = (x - ox) * sin[index] + (y - oy) * cos[index] + oy;

		return { nx, ny };
	}

	int get_new_pos_act(int act, int index)
	{
		if (act == PASS) return 0;
		auto xy = act_to_xy(act);
		auto new_xy = get_new_pos_xy(xy.first, xy.second, cx, cy, index);
		auto new_act = xy_to_act(new_xy.first, new_xy.second);
		return new_act;
	}

	string get_new_pos_str(string s, int index)
	{
		auto xy = act_to_xy(str_to_act(s));
		auto new_xy = get_new_pos_xy(xy.first, xy.second, cx, cy, index);
		string new_str = act_to_str(xy_to_act(new_xy.first, new_xy.second));
		return new_str;
	}

	vector<double> get_rotated_probs(vector<double> probs, int index)
	{
		vector<double> rotated_probs(ALL, 0);
		for (int act = PASS; act < ALL; act++)
		{
			rotated_probs[get_new_pos_act(act, index)] = probs[act];
		}
		return rotated_probs;
	}
}

int get_directory_num(string path)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	int cnt = 0;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				cnt++;
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return cnt;
}

int get_file_num(string path)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	int cnt = 0;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if (!(fileinfo.attrib & _A_SUBDIR))
			{
				cnt++;
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return cnt;
}

void delete_empty_directory(string path)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	int cnt = 0;
	string p;
	vector<string> all_directories;

	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (string(fileinfo.name) != "." && string(fileinfo.name) != "..")
				{
					all_directories.push_back(fileinfo.name);
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}

	for (int i = 0; i < all_directories.size(); i++)
	{
		if (get_file_num(path + all_directories[i]) < 5)
		{
			system((string("rm -r ") + path + all_directories[i]).c_str());
		}
	}
}

void delete_randomly(string path, int delete_cnt)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	int cnt = 0;
	string p;
	vector<string> all_files;

	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (string(fileinfo.name) != "." && string(fileinfo.name) != "..")
				{
					all_files.push_back(fileinfo.name);
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}

	random_shuffle(all_files.begin(), all_files.end());

	for (int i = 0; i < delete_cnt; i++)
	{
		//cout << (string("rmdir ") + path + all_files[i]) << endl;
		system((string("rm -r ") + path + all_files[i]).c_str());
	}
	//for (auto i : all_files)
	//{
	//	cout << i << endl;
	//}
}

void self_play_games(int thread_num = 12, double c_puct = 3.0,
	int simul_cnt = 1000, double virtual_loss = 0.6, int game_tot = 1,
	int random_turn = 6, int batch_size = BATCH_SIZE)
{
	NeuralNetwork net(string("./models/" + get_best_network() + ".pt"), true, batch_size);

	for (int game_cnt = 1; game_cnt <= game_tot; game_cnt++)
	{
		vector<GameField> gameFields(Rotate::RotateNum, GameField());

		auto game_directory = get_random_directory();
		system((string("mkdir .\\games\\") + game_directory).c_str());

		MCTS mcts(&net, thread_num, c_puct, simul_cnt, virtual_loss, ALL);
		GameField g;
		int turn_id = 0;

		vector<vector<double>> inputs;
		vector<double> value;
		vector<vector<double>> probs;

		// play a game
		while (g.referee() == unfinished)
		{
			// inputs
			for (int rotate_index = 0; rotate_index < Rotate::RotateNum; rotate_index++)
			{
				auto rotated_inputs = gameFields[rotate_index].get_gamefield_mat();
				inputs.emplace_back(rotated_inputs);
			}

			// get final move
			auto move_probs = mcts.get_action_probs(&g, 1);

			auto final_move = PASS;
			if (turn_id < random_turn)
			{
				vector<int> all_moves(ALL, PASS);
				for (int i = 0; i < ALL; i++)
				{
					all_moves[i] = i;
				}
				final_move = random_choice(all_moves, move_probs);

				// cout << "final move: " << final_move << endl;
			}
			else
			{
				final_move = best_choice(move_probs);
			}
			//cout << "------------" << turn_id << "------------" << endl;
			//print(move_probs);
			//cout << (g.current_color == black ? "black" : "white") << " play " << act_to_str(final_move) << endl;
			//cout << "value: " << -mcts.root->q_sa << endl;

			// host game field
			g.play(final_move);

			// rotated game fields play
			for (int rotate_index = 0; rotate_index < Rotate::RotateNum; rotate_index++)
			{
				auto rotated_move = Rotate::get_new_pos_act(final_move, rotate_index);
				//cout << "rotated move: " << rotated_move << " " << act_to_str(rotated_move) << endl;

				gameFields[rotate_index].play(rotated_move);
				//gameFields[rotate_index].print();
			}

			//cout << "1" << endl;

			// rotated act probs
			for (int rotate_index = 0; rotate_index < Rotate::RotateNum; rotate_index++)
			{
				auto rotated_probs = Rotate::get_rotated_probs(move_probs, rotate_index);
				probs.emplace_back(rotated_probs);
			}

			//g.print();

			// insert content
			value.insert(value.end(), Rotate::RotateNum, -mcts.root->q_sa);

			// mcts move
			mcts.update_with_move(final_move);
			turn_id++;
		}

		int game_status = g.referee();
		//cout << (game_status == black ? "black win" : "white win") << endl;

		write_file(value, (string("./games/") + game_directory + "/value.txt"));
		write_file(game_status, (string("./games/") + game_directory + "/winner.txt"));
		write_file(turn_id * 8, (string("./games/") + game_directory + "/length.txt"));
		write_file(inputs, (string("./games/") + game_directory + "/in.txt"));
		write_file(probs, (string("./games/") + game_directory + "/prob.txt"));
	}
}

void self_play_by_thread(int game_cnt)
{
	self_play_games(THREAD_NUM, SELFPLAY_CPUCT, SELFPLAY_SIMUL_NUM, VIRTUAL_LOSS, game_cnt, SELFPLAY_RANDOM_TURN);
}

int hold_contest_between_nets(string old_net_index, string new_net_index, int game_num, int batch_size = BATCH_SIZE)
{
	NeuralNetwork old_net(string("./models/" + old_net_index + ".pt"), true, batch_size);
	NeuralNetwork new_net(string("./models/" + new_net_index + ".pt"), true, batch_size);

	int win_cnt = 0;
	for (int game_cnt = 0; game_cnt < game_num; game_cnt++)
	{
		bool old_net_first = (rd() % 2) ? true : false;

		MCTS old_tree(&old_net, THREAD_NUM, CONTEST_CPUCT, CONTEST_SIMUL_NUM, VIRTUAL_LOSS, ALL);
		MCTS new_tree(&new_net, THREAD_NUM, CONTEST_CPUCT, CONTEST_SIMUL_NUM, VIRTUAL_LOSS, ALL);

		GameField g;
		int turn_id = 0;
		while (g.referee() == unfinished)
		{
			int move = PASS;
			vector<double> move_probs;

			if ((turn_id % 2 == 0 && old_net_first) ||
				(turn_id % 2 == 1 && !old_net_first))
			{
				move_probs = old_tree.get_action_probs(&g, 1);
			}
			else
			{
				move_probs = new_tree.get_action_probs(&g, 1);
			}

			if (turn_id < CONTEST_RANDOM_TURN)
			{
				vector<int> all_moves(ALL, PASS);
				for (int i = 0; i < ALL; i++)
				{
					all_moves[i] = i;
				}
				move = random_choice(all_moves, move_probs);
			}
			else
			{
				move = best_choice(move_probs);
			}

			g.play(move);
			old_tree.update_with_move(move);
			new_tree.update_with_move(move);

			turn_id++;
		}

		auto status = g.referee();
		if ((status == white && old_net_first) ||
			(status == black && !old_net_first))
		{
			win_cnt++;
		}
	}
	return win_cnt;
}

void net_contest_by_thread(string old_net_index, string new_net_index, int game_num, shared_ptr<promise<int>> win_cnt, int batch_size = BATCH_SIZE)
{
	win_cnt->set_value(hold_contest_between_nets(old_net_index, new_net_index, game_num, batch_size));
}

double get_winning_rate_multi_thread(string best, string newest)
{
	int sum = 0;
	int contest_num_per_thread = CONTEST_GAME_NUM / (THREAD_NUM / 2);

	vector<shared_ptr<promise<int>>> win_cnt_promises;
	vector<future<int>> win_cnt_ftrs;
	vector<thread> thread_vector;

	for (int i = 0; i < (THREAD_NUM / 2); i++)
	{
		auto finished_promise = std::make_shared<std::promise<int>>();
		win_cnt_promises.emplace_back(finished_promise);

		win_cnt_ftrs.emplace_back(finished_promise->get_future());
		thread_vector.emplace_back(
			std::move(
				std::thread(
					net_contest_by_thread, best, newest, contest_num_per_thread, finished_promise, BATCH_SIZE
				)
			)
		);
	}
	for (auto& ftr: win_cnt_ftrs)
	{
		ftr.wait();
	}
	for (auto& ftr : win_cnt_ftrs)
	{
		sum += ftr.get();
	}
	for (auto& th : thread_vector)
	{
		th.join();
	}
	cout << "win: " << sum << endl;
	return 1.0 * sum / CONTEST_GAME_NUM;
}

double get_winning_rate(string best, string newest)
{
	int win_cnt = hold_contest_between_nets(best, newest, CONTEST_GAME_NUM);
	cout << "win: " << win_cnt << endl;
	return 1.0 * win_cnt / CONTEST_GAME_NUM;
}

void train(int game_tot = GAME_PER_TRAIN, int epoch_tot = ITERATION_NUM, int remove_game_cnt = REMOVE_GAME_CNT)
{
	delete_empty_directory("./games/");
	for (int epoch = 1; epoch <= epoch_tot; epoch++)
	{
		bool passed = false;

		auto game_left = game_tot - get_directory_num("./games/") + 2;
		if (game_left > 0)
		{
			vector<thread> thread_vector;
			int game_cnt_per_thread = game_left / THREAD_NUM;
			for (int thread_id = 0; thread_id < THREAD_NUM; thread_id++)
			{
				thread_vector.push_back(thread(self_play_by_thread, game_cnt_per_thread));
			}
			for (int thread_id = 0; thread_id < THREAD_NUM; thread_id++)
			{
				thread_vector[thread_id].join();
			}
		}
		system("bat\\train.bat");
		if (epoch % CHECKPOINT_FEQ == 0)
		{
			cout << "-----check point-----" << endl;
			auto newest_index = get_newest_network();
			auto best_index = get_best_network();

			auto winning_rate = get_winning_rate_multi_thread(best_index, newest_index);
			cout << "winning rate: " << winning_rate << endl;
			write_file(winning_rate, "./winning_rate.txt");

			if (winning_rate > NET_PASS_THRESHOLD)
			{
				write_file(get_newest_network(), string("./index/best.txt"));
				passed = true;
			}
		}

		if (passed)
		{
			delete_randomly(".\\games\\", remove_game_cnt * 2);
		}
		else
		{
			delete_randomly(".\\games\\", remove_game_cnt);
		}
	}
}

int main()
{
	try 
	{
		train();
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}