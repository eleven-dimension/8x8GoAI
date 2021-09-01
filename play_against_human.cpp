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

void play_game_against_human(int thread_num = 12, double c_puct = 5.0,
	int simul_cnt = 1000, double virtual_loss = 0.6, int game_tot = 1,
	int batch_size = 512, bool jws_first = true)
{
	NeuralNetwork net(string("./models/" + get_best_network() + ".pt"), true, batch_size);

	for (int game_cnt = 1; game_cnt <= game_tot; game_cnt++)
	{

		MCTS mcts(&net, thread_num, c_puct, simul_cnt, virtual_loss, ALL);
		GameField g;
		int turn_id = 0;

		vector<double> move_probs;
		auto final_move = PASS;

		while (g.referee() == unfinished)
		{
			if ((turn_id % 2 == 0 && jws_first) ||
				(turn_id % 2 != 0 && jws_first == false))
			{
				move_probs = mcts.get_action_probs(&g, 1);
				final_move = best_choice(move_probs);

				print(move_probs);
				cout << (g.current_color == black ? "black" : "white") << " play " << act_to_str(final_move) << endl;
				cout << "value: " << -mcts.root->q_sa << endl;
			}
			else
			{
				cout << "HUMAN: ";
				string human_move_str;
				cin >> human_move_str;
				final_move = str_to_act(human_move_str);
			}

			g.play(final_move);
			g.print();

			mcts.update_with_move(final_move);
			turn_id++;
		}

		int game_status = g.referee();
		cout << (game_status == black ? "black win" : "white win") << endl;
	}
}

int main()
{
	play_game_against_human();
}