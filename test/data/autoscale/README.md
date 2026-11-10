# autoScale example data

Order parameter (relative size of the largest cluster) of two-dimensional site percolation
for system sizes L = 128, 256 and 512. Columns: occupation probability `x`, order parameter
`y` and its standard error `dy`.

Copied unmodified from the `ORDER_PARAMETER` directory of
[omelchert/autoScale](https://github.com/omelchert/autoScale) at commit
`a0e98da2098c2b0561edb2c54b9435f31ff884ec`, licensed under the MIT license (see
`LICENSE.md`).

The reference results used in `test/autoscale_reference.jl` come from the same commit:
`README.md` and `scaled_L512_256_128.out` (S at the best fit for several `-xr` ranges) and
`error.out` (best fit and S+1 error bars for `-xr -1 1`).
