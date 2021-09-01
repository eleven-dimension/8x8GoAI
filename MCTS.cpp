#include <cmath>
#include <cfloat>
#include <numeric>
#include <iostream>

#include "MCTS.h"

// TreeNode
TreeNode::TreeNode()
    : parent(nullptr),
    is_leaf(true),
    virtual_loss(0),
    n_visited(0),
    p_sa(0),
    q_sa(0) {}

TreeNode::TreeNode(TreeNode* parent, double p_sa, unsigned int action_size)
    : parent(parent),
    children(action_size, nullptr),
    is_leaf(true),
    virtual_loss(0),
    n_visited(0),
    q_sa(0),
    p_sa(p_sa) {}

TreeNode::TreeNode(
    const TreeNode& node) {  // because automic<>, define copy function
  // struct
    this->parent = node.parent;
    this->children = node.children;
    this->is_leaf = node.is_leaf;

    this->n_visited.store(node.n_visited.load());
    this->p_sa = node.p_sa;
    this->q_sa = node.q_sa;

    this->virtual_loss.store(node.virtual_loss.load());
}

TreeNode& TreeNode::operator=(const TreeNode& node) {
    if (this == &node) {
        return *this;
    }

    // struct
    this->parent = node.parent;
    this->children = node.children;
    this->is_leaf = node.is_leaf;

    this->n_visited.store(node.n_visited.load());
    this->p_sa = node.p_sa;
    this->q_sa = node.q_sa;
    this->virtual_loss.store(node.virtual_loss.load());

    return *this;
}

unsigned int TreeNode::select(double c_puct, double c_virtual_loss) {
    double best_value = -DBL_MAX;
    unsigned int best_move = 0;
    TreeNode* best_node;

    for (unsigned int i = 0; i < this->children.size(); i++) {
        // empty node
        if (children[i] == nullptr) {
            continue;
        }

        unsigned int sum_n_visited = this->n_visited.load() + 1;
        double cur_value =
            children[i]->get_value(c_puct, c_virtual_loss, sum_n_visited);
        if (cur_value > best_value) {
            best_value = cur_value;
            best_move = i;
            best_node = children[i];
        }
    }

    // add vitural loss
    best_node->virtual_loss++;

    return best_move;
}

void TreeNode::expand(const std::vector<double>& action_priors) {
    {
        // get lock
        std::lock_guard<std::mutex> lock(this->lock);

        if (this->is_leaf) {
            unsigned int action_size = this->children.size();

            for (unsigned int i = 0; i < action_size; i++) {
                // illegal action
                if (abs(action_priors[i] - 0) < FLT_EPSILON) {
                    continue;
                }
                this->children[i] = new TreeNode(this, action_priors[i], action_size);
            }

            // not leaf
            this->is_leaf = false;
        }
    }
}

void TreeNode::backup(double value) {
    // If it is not root, this node's parent should be updated first
    if (this->parent != nullptr) {
        this->parent->backup(-value);
    }

    // remove vitural loss
    this->virtual_loss--;

    // update n_visited
    unsigned int n_visited = this->n_visited.load();
    this->n_visited++;

    // update q_sa
    {
        std::lock_guard<std::mutex> lock(this->lock);
        this->q_sa = (n_visited * this->q_sa + value) / (n_visited + 1);
    }
}

double TreeNode::get_value(double c_puct, double c_virtual_loss,
    unsigned int sum_n_visited) const {
    // u
    auto n_visited = this->n_visited.load();
    double u = (c_puct * this->p_sa * sqrt(sum_n_visited) / (1 + n_visited));

    // virtual loss
    double virtual_loss = c_virtual_loss * this->virtual_loss.load();
    // int n_visited_with_loss = n_visited - virtual_loss;

    if (n_visited <= 0) {
        return u;
    }
    else {
        return u + (this->q_sa * n_visited - virtual_loss) / n_visited;
    }
}

// MCTS
MCTS::MCTS(NeuralNetwork* neural_network, unsigned int thread_num, double c_puct,
    unsigned int num_mcts_sims, double c_virtual_loss,
    unsigned int action_size)
    : neural_network(neural_network),
    thread_pool(new ThreadPool(thread_num)),
    c_puct(c_puct),
    num_mcts_sims(num_mcts_sims),
    c_virtual_loss(c_virtual_loss),
    action_size(action_size),
    root(new TreeNode(nullptr, 1., action_size), MCTS::tree_deleter) {}

void MCTS::update_with_move(int last_action) {
    auto old_root = this->root.get();

    // reuse the child tree
    if (last_action >= 0 && old_root->children[last_action] != nullptr) {
        // unlink
        TreeNode* new_node = old_root->children[last_action];
        old_root->children[last_action] = nullptr;
        new_node->parent = nullptr;

        this->root.reset(new_node);
    }
    else {
        this->root.reset(new TreeNode(nullptr, 1., this->action_size));
    }
}

void MCTS::tree_deleter(TreeNode* t) {
    if (t == nullptr) {
        return;
    }

    // remove children
    for (unsigned int i = 0; i < t->children.size(); i++) {
        if (t->children[i]) {
            tree_deleter(t->children[i]);
        }
    }

    // remove self
    delete t;
}

std::vector<double> MCTS::get_action_probs(GameField* g, double temp) {
    // submit simulate tasks to thread_pool
    std::vector<std::future<void>> futures;

    for (unsigned int i = 0; i < this->num_mcts_sims; i++) {
        // copy gomoku
        auto game = std::make_shared<GameField>(*g);
        auto future =
            this->thread_pool->commit(std::bind(&MCTS::simulate, this, game, true));

        // future can't copy
        futures.emplace_back(std::move(future));
    }

    // wait simulate
    for (unsigned int i = 0; i < futures.size(); i++) {
        futures[i].wait();
    }

    // std::cout << "simulation ends" << std::endl;

    // calculate probs
    std::vector<double> action_probs(ALL, 0);
    const auto& children = this->root->children;

    // greedy
    if (temp - 1e-3 < FLT_EPSILON) {
        unsigned int max_count = 0;
        unsigned int best_action = 0;

        for (unsigned int i = 0; i < children.size(); i++) {
            if (children[i] && children[i]->n_visited.load() > max_count) {
                max_count = children[i]->n_visited.load();
                best_action = i;
            }
        }

        action_probs[best_action] = 1.;
        return action_probs;

    }
    else {
        // explore
        double sum = 0;
        for (unsigned int i = 0; i < children.size(); i++) {
            if (children[i] && children[i]->n_visited.load() > 0) {
                action_probs[i] = pow(children[i]->n_visited.load(), 1 / temp);
                sum += action_probs[i];
            }
        }

        // renormalization
        std::for_each(action_probs.begin(), action_probs.end(),
            [sum] (double& x) { x /= sum; });

        return action_probs;
    }
}

void MCTS::simulate(std::shared_ptr<GameField> g, bool explore)
{
    auto node = this->root.get();

    while (true)
    {
        if (node->is_leaf) break;
        auto action = node->select(this->c_puct, this->c_virtual_loss);
        g->play(action);
        node = node->children[action];
    }

    auto status = g->referee();
    double value = 0;

    if (status == unfinished)
    {
        //std::pair<std::vector<float>, float> net_output;
        //try 
        //{
        //    auto net_ouput = neural_network->get_net_output(g);
        //}
        //catch (std::runtime_error& e)
        //{
        //    std::cout << e.what() << std::endl;
        //    exit(1);
        //}
        auto future = this->neural_network->commit(g.get());
        auto result = future.get();
        auto net_pri_probs = std::move(result[0]);

        value = result[1][0];
        //std::cout << "--------" << std::endl;
        //for (auto i : net_pri_probs)
        //{
        //    std::cout << i << std::endl;
        //}
        //std::cout << "--------" << std::endl;

        auto pri_probs = net_pri_probs;
        auto legal_moves_mask = g->valid_moves_mask(g->current_color);
        double sum = 0;

        for (int i = 0; i < legal_moves_mask.size(); i++)
        {
            if (legal_moves_mask[i] == 1)
            {
                sum += net_pri_probs[i];
            }
            else pri_probs[i] = 0;
        }

        std::for_each(pri_probs.begin(), pri_probs.end(), [sum] (double& x) { x /= sum; });

        auto long_noise_prob = std::vector<double>(ALL, 0);
        if (explore)
        {
            int valid_cnt = std::accumulate(legal_moves_mask.begin(), legal_moves_mask.end(), 0);
            auto noise_prob = get_noise(valid_cnt);

            int noise_ptr = 0;
            for (int i = 0; i < ALL; i++)
            {
                if (legal_moves_mask[i] == 1)
                {
                    long_noise_prob[i] = noise_prob[noise_ptr++];
                }
            }
            for (int i = 0; i < ALL; i++)
            {
                pri_probs[i] = 0.8 * pri_probs[i] + 0.2 * long_noise_prob[i];
            }
        }

        node->expand(pri_probs);
    }
    else
    {
        auto winner = status;
        value = (winner == g->current_color ? 1 : -1);
    }
    node->backup(-value);
}