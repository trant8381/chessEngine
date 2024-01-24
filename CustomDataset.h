#include <torch/torch.h>

template <typename Data = torch::Tensor, typename Target = torch::Tensor,
          typename Mask = torch::Tensor>
struct Example {
  using DataType = Data;
  using TargetType = Target;
  using MaskType = Mask;

  Data data;
  Target target;
  Mask mask;

  Example() = default;
  Example(Data data, Target target, Mask mask)
      : data(std::move(data)), target(std::move(target)),
        mask(std::move(mask)) {}
};

template <typename ExampleType>
struct Stack : public torch::data::transforms::Collation<ExampleType> {
  ExampleType apply_batch(std::vector<ExampleType> examples) override {
    std::vector<torch::Tensor> xs, ys, masks;
    xs.reserve(examples.size());
    ys.reserve(examples.size());
    masks.reserve(examples.size());
    for (auto &example : examples) {
      xs.push_back(std::move(example.data));
      ys.push_back(std::move(example.target));
      masks.push_back(std::move(example.mask));
    }
    return {torch::stack(xs), torch::stack(ys), torch::stack(masks)};
  }
};

using Example3 = Example<torch::Tensor, torch::Tensor, torch::Tensor>;
class CustomDataset : public torch::data::Dataset<CustomDataset, Example3> {
public:
    torch::Tensor half1;
    torch::Tensor half2;
    torch::Tensor evals;
    CustomDataset(std::vector<torch::Tensor> _half1, std::vector<float> _evals, std::vector<torch::Tensor> _half2) {
        half1 = torch::stack(_half1).cuda();
        half2 = torch::stack(_half2).cuda();
        evals = torch::from_blob(_evals.data(), {static_cast<long>(_evals.size())}, torch::TensorOptions().dtype(torch::kFloat)).cuda();
    }

    torch::optional<size_t> size() const override {
        return half1.size(0);
    }

    Example3 get(size_t index) override {
        return { half1[index], evals[index], evals[index] };
    }
};