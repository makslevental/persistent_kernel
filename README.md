# Kernel launch overheads

All times are in seconds. Driver Version: 510.39.01

## cuStreamWaitValue32

| Task   |    mean     | median      | std         | min         | max         |
|--------|:-----------:|-------------|-------------|-------------|-------------|
| Launch | 3.21609e-08 | 3.00352e-08 | 6.02631e-09 | 2.99769e-08 | 1.42195e-06 | 
| Signal | 1.47105e-07 | 1.50001e-07 | 1.71373e-08 | 1.29978e-07 | 3.14601e-06 |  

## Spin

| Task   |    mean     | median      | std         | min         | max         |
|--------|:-----------:|-------------|-------------|-------------|-------------|
| Launch | 1.48487e-06 | 1.45304e-06 | 1.38428e-07 | 1.33197e-06 | 8.97702e-06 | 
| Signal | 6.21179e-06 | 5.89102e-06 | 1.04506e-05 | 3.39991e-07 | 0.000749292 | 
## Takeaway

Using `cuStreamWaitValue32` you can pause CUDA streams in ~30ns and unpause them in ~150ns.
This is about an order of magnitude faster than conventional kernel launch, sitting at 2.6μs, at the minimum.