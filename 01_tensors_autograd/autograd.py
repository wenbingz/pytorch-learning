import torch


def main() -> None:
    torch.manual_seed(0)

    # y = (w*x + b)^2
    x = torch.tensor(3.0)
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(-1.0, requires_grad=True)

    y = (w * x + b) ** 2
    y.backward()

    print("y:", y.item())
    print("dw:", w.grad.item())  # 2*(w*x+b)*x
    print("db:", b.grad.item())  # 2*(w*x+b)

    # gradients accumulate by default
    y2 = (w * x + b) ** 2
    y2.backward()
    print("dw accumulated:", w.grad.item())
    print("db accumulated:", b.grad.item())

    # reset grads (preferred over .zero_() for many optimizers)
    w.grad = None
    b.grad = None

    # one manual SGD step (just to see a parameter update)
    lr = 0.1
    with torch.no_grad():
        grad_w = 2 * (w * x + b) * x
        w -= lr * grad_w
    print("w after step:", w.item())


if __name__ == "__main__":
    main()

