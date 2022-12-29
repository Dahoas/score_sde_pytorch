import torch

inputs = torch.arange(4)
print("inputs", inputs)
fft_out = torch.fft.fft(inputs, norm="ortho")
print("fft out", fft_out)
ifft_out = torch.fft.ifft(fft_out, norm="ortho")
print("ifft out", ifft_out)

inputs = torch.rand(128, 3, 32, 32)
print("inputs", inputs.size(), inputs.dtype)
#rfft_out = torch.fft.rfft(inputs, norm="ortho", dim=1)
fft_out = torch.fft.fft(inputs, norm="ortho", dim=-2)
#fft_out_2 = torch.fft.fft(inputs, norm="ortho", dim=-2)
squeezed_fft_out = torch.fft.fft(inputs.view(128, 3, -1), norm="ortho", dim=1).view(128, 3, 32, 32)
print(torch.sum(fft_out - squeezed_fft_out))
#print("rfft", rfft_out.size(), rfft_out.dtype)
print("fft", fft_out.size(), fft_out.dtype)
print("squeezed_fft_out", squeezed_fft_out.size())

fft2_out = torch.fft.fftn(inputs, norm="ortho", dim=[-1, 2])
print("fft2", fft2_out.size())