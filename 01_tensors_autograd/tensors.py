import torch


def main() -> None:
    torch.manual_seed(0)

    a = torch.arange(6).reshape(2, 3)
    b = torch.tensor([10, 20, 30])  # shape (3,)
    c = a + b  # broadcast (2, 3) + (3,) -> (2, 3)

    print("a:\n", a)
    print("b:", b, "shape", tuple(b.shape))
    print("c:\n", c)

    print("a[0]:", a[0])
    print("a[:, 1]:", a[:, 1])

    x = torch.randn(4, dtype=torch.float32)
    y = x.to(torch.float16)
    print("x dtype:", x.dtype, "y dtype:", y.dtype)


if __name__ == "__main__":
    main()

