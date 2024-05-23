import torch

def print_bad_elements(t1, t2):
    num0 = t1.shape[0]
    num1 = t1.shape[1]
    for i in range(num0):
        for j in range(num1):
            if t1[i, j] != t2[i, j]:
                print(f"Bad element [{i}, {j}]: {t1[i, j]} {t2[i, j]}")
def compare_tensors(t1, t2):
    for i in range(t1.shape[0]):
        t1p = t1[i, 0, :, :]
        t2p = t2[i, 0, :, :]
        OK = True
        if not torch.all( t1p == t2p ).item():
            print(f"Element-wise fail for {i}.")
            OK = False

        if not torch.all(torch.eq(t1p, t2p)).item():
            print(f"Equality fail for {i}.")
            OK = False

        if not OK:
            print_bad_elements(t1p, t2p)


def main(args):
    print("Hello, World!")
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # Load the three tensors from disk
    file_path1 = '/Users/metzler/Work/PyCharmProjects/PlayNeuralNetworks/Results38/'
    file_path2 = '/Users/metzler/Work/PyCharmProjects/PlayNeuralNetworks/Results39/'

    file_prefix = 'validate'
    fname_x1 = file_path1 + file_prefix + '_x.tar'  # Adjust the filenames accordingly
    fname_y1 = file_path1 + file_prefix + '_y.tar'
    fname_p1 = file_path1 + file_prefix + '_p.tar'
    fname_x2 = file_path2 + file_prefix + '_x.tar'  # Adjust the filenames accordingly
    fname_y2 = file_path2 + file_prefix + '_y.tar'
    fname_p2 = file_path2 + file_prefix + '_p.tar'

    X_validate1 = torch.load(fname_x1, map_location=device).view(-1, 1, 256, 256)
    X_validate2 = torch.load(fname_x2, map_location=device).view(-1, 1, 256, 256)
    print(f"X_validate: {X_validate1.shape} {X_validate2.shape}")
    y_validate1 = torch.load(fname_y1, map_location=device)
    y_validate2 = torch.load(fname_y2, map_location=device)
    print(f"y_validate: {y_validate1.shape} {y_validate2.shape}")
    y_pred1 = torch.load(fname_p1, map_location=device)
    y_pred2 = torch.load(fname_p2, map_location=device)
    print(f"y_pred: {y_pred1.shape} {y_pred2.shape}")

    print("Comparing x")
    compare_tensors(X_validate1, X_validate2)

    print("Comparing y")
    compare_tensors(y_validate1, y_validate2)

    print("Comparing p")
    compare_tensors(y_pred1, y_pred2)

if __name__ == "__main__":
    main(None)
