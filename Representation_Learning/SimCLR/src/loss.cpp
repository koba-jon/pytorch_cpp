#include <iostream>
#include <string>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "loss.hpp"
#include "losses.hpp"

// Define Namespace
namespace F = torch::nn::functional;


// -----------------------------------
// class{Loss} -> operator
// -----------------------------------
torch::Tensor Loss::operator()(torch::Tensor z1, torch::Tensor z2, const float tau){

    long int N;
    torch::Tensor z, logits, targets, loss;

    N = z1.size(0);
    z1 = F::normalize(z1, F::NormalizeFuncOptions().p(2).dim(1));
    z2 = F::normalize(z2, F::NormalizeFuncOptions().p(2).dim(1));

    z = torch::cat({z1, z2}, /*dim=*/0);
    logits = torch::mm(z, z.t()) / tau;
    logits.diagonal(/*offset=*/0, /*dim1=*/0, /*dim2=*/1).fill_(-1e9);

    targets = torch::arange(0, 2 * N).to(torch::kLong).to(z1.device());
    targets = torch::where(targets < N, targets + N, targets - N);
    loss = F::cross_entropy(logits, targets);

    return loss;

}
