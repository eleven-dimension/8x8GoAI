#include <iostream>
#include <vector>
#include <string>

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/api/include/torch/torch.h>

#include "libtorch.h"
#include "GameField.h"
#include "MCTS.h"

#include <chrono>

using namespace std;
using namespace chrono;

int main()
{
	try {
		//auto network(
		//	std::make_shared<torch::jit::script::Module>(
		//		torch::jit::load("D:\\Project\\py\\torchGo\\models\\checkpoint.pt")
		//		)
		//);
		////cout << network << endl;
		//std::vector<torch::jit::IValue> inputs = { torch::ones({ 2, 3, 8, 8 }) };
		////cout << inputs[0] << endl;

		//auto result = network->forward(inputs).toTuple();
		////cout << result << endl;
		//torch::Tensor p_batch = result->elements()[0]
		//	.toTensor()
		//	.exp()
		//	.toType(torch::kFloat32)
		//	.to(at::kCPU);
		//torch::Tensor v_batch =
		//	result->elements()[1].toTensor().toType(torch::kFloat32).to(at::kCPU);
		//cout << p_batch << endl;
		//cout << v_batch << endl;
		
		GameField g;
		NeuralNetwork net("./models/checkpoint.pt", true, 64);

		MCTS mcts(&net, 12, 5, 800, 3, 65);

		g.print();

		auto start = system_clock::now();
		auto p = mcts.get_action_probs(&g, 1);
		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		cout << double(duration.count()) * microseconds::period::num / microseconds::period::den
			<< " seconds" << endl;

		for (auto i : p)
		{
			cout << i << endl;
		}
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}