# Benchmark Results (L1–L5)

## GPT-5.5

### L1

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_cpdag_llm | 0.827 | 1.6 | 0.873 | 0.810 |
| llm_raw | 0.824 | 1.8 | 0.881 | 0.907 |
| pc_greedy | 0.802 | 1.9 | 0.848 | 0.774 |
| llm_stats | 0.783 | 2.1 | 0.848 | 0.852 |
| llm_stats_cpdag_greedy | 0.528 | 4.0 | 0.609 | 0.613 |

### L2

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| llm_stats | 0.704 | 4.1 | 0.782 | 0.725 |
| llm_raw | 0.734 | 4.2 | 0.792 | 0.743 |
| pc_cpdag_llm | 0.666 | 4.4 | 0.760 | 0.528 |
| pc_greedy | 0.613 | 4.9 | 0.740 | 0.528 |
| llm_stats_cpdag_greedy | 0.425 | 8.0 | 0.529 | 0.536 |

### L3

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.697 | 5.5 | 0.779 | 0.629 |
| pc_cpdag_llm | 0.688 | 5.5 | 0.779 | 0.626 |
| llm_raw | 0.705 | 6.0 | 0.772 | 0.753 |
| llm_stats | 0.628 | 7.8 | 0.661 | 0.593 |
| llm_stats_cpdag_greedy | 0.423 | 8.9 | 0.509 | 0.433 |

### L4

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.690 | 6.2 | 0.775 | 0.576 |
| pc_cpdag_llm | 0.679 | 6.4 | 0.775 | 0.557 |
| llm_raw | 0.686 | 7.1 | 0.713 | 0.673 |
| llm_stats | 0.578 | 9.6 | 0.646 | 0.645 |
| llm_stats_cpdag_greedy | 0.306 | 12.9 | 0.386 | 0.347 |

### L5

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.642 | 8.1 | 0.759 | 0.554 |
| pc_cpdag_llm | 0.646 | 8.1 | 0.759 | 0.553 |
| llm_raw | 0.537 | 12.2 | 0.606 | 0.582 |
| llm_stats | 0.508 | 12.8 | 0.586 | 0.552 |
| llm_stats_cpdag_greedy | 0.372 | 17.0 | 0.464 | 0.345 |

### Avg L1–L5

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_cpdag_llm | 0.701 | 5.2 | 0.789 | 0.615 |
| pc_greedy | 0.689 | 5.3 | 0.780 | 0.612 |
| llm_raw | 0.697 | 6.3 | 0.753 | 0.732 |
| llm_stats | 0.640 | 7.3 | 0.704 | 0.673 |
| llm_stats_cpdag_greedy | 0.411 | 10.2 | 0.499 | 0.455 |

## GPT-5.4-mini

### L1

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.802 | 1.9 | 0.848 | 0.774 |
| pc_cpdag_llm | 0.410 | 4.4 | 0.817 | 0.632 |
| llm_stats_cpdag_greedy | 0.138 | 6.0 | 0.138 | 0.175 |
| llm_raw | 0.240 | 7.6 | 0.480 | 0.342 |
| llm_stats | 0.252 | 11.4 | 0.404 | 0.483 |

### L2

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.613 | 4.9 | 0.740 | 0.528 |
| pc_cpdag_llm | 0.414 | 6.4 | 0.740 | 0.406 |
| llm_stats_cpdag_greedy | 0.025 | 9.5 | 0.025 | 0.050 |
| llm_raw | 0.090 | 14.9 | 0.266 | 0.174 |
| llm_stats | 0.083 | 15.8 | 0.286 | 0.108 |

### L3

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.697 | 5.5 | 0.779 | 0.629 |
| pc_cpdag_llm | 0.465 | 8.1 | 0.763 | 0.489 |
| llm_stats_cpdag_greedy | 0.038 | 12.0 | 0.038 | 0.028 |
| llm_stats | 0.082 | 19.0 | 0.278 | 0.121 |
| llm_raw | 0.167 | 24.6 | 0.330 | 0.271 |

### L4

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.690 | 6.2 | 0.775 | 0.576 |
| pc_cpdag_llm | 0.505 | 9.0 | 0.764 | 0.518 |
| llm_stats_cpdag_greedy | 0.033 | 14.4 | 0.033 | 0.000 |
| llm_stats | 0.076 | 25.8 | 0.155 | 0.087 |
| llm_raw | 0.156 | 27.1 | 0.265 | 0.279 |

### L5

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.642 | 8.1 | 0.759 | 0.554 |
| pc_cpdag_llm | 0.349 | 12.9 | 0.746 | 0.364 |
| llm_stats_cpdag_greedy | 0.028 | 16.5 | 0.028 | 0.058 |
| llm_stats | 0.021 | 21.4 | 0.099 | 0.039 |
| llm_raw | 0.070 | 37.5 | 0.236 | 0.146 |

### Avg L1–L5

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.689 | 5.3 | 0.780 | 0.612 |
| pc_cpdag_llm | 0.429 | 8.2 | 0.766 | 0.482 |
| llm_stats_cpdag_greedy | 0.053 | 11.7 | 0.053 | 0.062 |
| llm_stats | 0.103 | 18.6 | 0.244 | 0.168 |
| llm_raw | 0.145 | 22.4 | 0.315 | 0.242 |

## Sonnet-4.6

### L1

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.802 | 1.9 | 0.848 | 0.774 |
| pc_cpdag_llm | 0.742 | 2.4 | 0.825 | 0.786 |
| llm_stats_cpdag_greedy | 0.418 | 5.2 | 0.744 | 0.607 |
| llm_raw | 0.426 | 6.0 | 0.514 | 0.558 |
| llm_stats | 0.164 | 9.1 | 0.498 | 0.321 |

### L2

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.613 | 4.9 | 0.740 | 0.528 |
| pc_cpdag_llm | 0.585 | 5.1 | 0.740 | 0.528 |
| llm_stats_cpdag_greedy | 0.374 | 8.8 | 0.547 | 0.392 |
| llm_raw | 0.232 | 11.0 | 0.433 | 0.311 |
| llm_stats | 0.187 | 12.7 | 0.398 | 0.321 |

### L3

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.697 | 5.5 | 0.779 | 0.629 |
| pc_cpdag_llm | 0.629 | 6.4 | 0.779 | 0.563 |
| llm_stats_cpdag_greedy | 0.095 | 12.5 | 0.124 | 0.129 |
| llm_raw | 0.319 | 13.8 | 0.446 | 0.377 |
| llm_stats | 0.233 | 14.4 | 0.316 | 0.269 |

### L4

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.690 | 6.2 | 0.775 | 0.576 |
| pc_cpdag_llm | 0.606 | 7.5 | 0.786 | 0.594 |
| llm_stats_cpdag_greedy | 0.217 | 14.3 | 0.212 | 0.195 |
| llm_raw | 0.294 | 15.2 | 0.363 | 0.273 |
| llm_stats | 0.206 | 18.6 | 0.311 | 0.272 |

### L5

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.642 | 8.1 | 0.759 | 0.554 |
| pc_cpdag_llm | 0.569 | 9.2 | 0.756 | 0.538 |
| llm_stats_cpdag_greedy | 0.237 | 18.0 | 0.297 | 0.251 |
| llm_raw | 0.252 | 21.2 | 0.315 | 0.358 |
| llm_stats | 0.127 | 22.0 | 0.235 | 0.188 |

### Avg L1–L5

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.689 | 5.3 | 0.780 | 0.612 |
| pc_cpdag_llm | 0.626 | 6.1 | 0.777 | 0.602 |
| llm_stats_cpdag_greedy | 0.269 | 12.0 | 0.382 | 0.314 |
| llm_raw | 0.314 | 12.3 | 0.430 | 0.390 |
| llm_stats | 0.183 | 15.4 | 0.350 | 0.273 |

## Haiku-4.5

### L1

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.802 | 1.9 | 0.848 | 0.774 |
| pc_cpdag_llm | 0.599 | 3.2 | 0.811 | 0.761 |
| llm_stats_cpdag_greedy | 0.375 | 5.9 | 0.679 | 0.613 |
| llm_stats | 0.219 | 6.9 | 0.523 | 0.342 |
| llm_raw | 0.192 | 9.0 | 0.401 | 0.333 |

### L2

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.613 | 4.9 | 0.740 | 0.528 |
| pc_cpdag_llm | 0.369 | 6.6 | 0.717 | 0.401 |
| llm_stats_cpdag_greedy | 0.377 | 8.4 | 0.554 | 0.503 |
| llm_stats | 0.183 | 10.6 | 0.296 | 0.249 |
| llm_raw | 0.160 | 13.5 | 0.273 | 0.227 |

### L3

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.697 | 5.5 | 0.779 | 0.629 |
| pc_cpdag_llm | 0.502 | 7.6 | 0.779 | 0.481 |
| llm_stats_cpdag_greedy | 0.321 | 14.4 | 0.502 | 0.339 |
| llm_stats | 0.051 | 16.9 | 0.168 | 0.063 |
| llm_raw | 0.085 | 19.4 | 0.292 | 0.095 |

### L4

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.690 | 6.2 | 0.775 | 0.576 |
| pc_cpdag_llm | 0.447 | 9.6 | 0.763 | 0.527 |
| llm_stats | 0.075 | 19.5 | 0.156 | 0.044 |
| llm_stats_cpdag_greedy | 0.258 | 19.8 | 0.450 | 0.275 |
| llm_raw | 0.103 | 22.5 | 0.217 | 0.134 |

### L5

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.642 | 8.1 | 0.759 | 0.554 |
| pc_cpdag_llm | 0.465 | 10.8 | 0.753 | 0.481 |
| llm_stats | 0.026 | 17.2 | 0.131 | 0.031 |
| llm_stats_cpdag_greedy | 0.113 | 19.4 | 0.222 | 0.159 |
| llm_raw | 0.047 | 27.4 | 0.230 | 0.081 |

### Avg L1–L5

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.689 | 5.3 | 0.780 | 0.612 |
| pc_cpdag_llm | 0.479 | 7.6 | 0.766 | 0.534 |
| llm_stats_cpdag_greedy | 0.287 | 13.5 | 0.484 | 0.380 |
| llm_stats | 0.111 | 14.2 | 0.255 | 0.146 |
| llm_raw | 0.113 | 18.9 | 0.280 | 0.167 |

## Gemini-3-Flash

### L1

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.802 | 1.9 | 0.848 | 0.774 |
| pc_cpdag_llm | 0.709 | 2.4 | 0.848 | 0.810 |
| llm_stats_cpdag_greedy | 0.579 | 4.2 | 0.728 | 0.683 |
| llm_stats | 0.345 | 6.1 | 0.645 | 0.617 |
| llm_raw | 0.414 | 6.1 | 0.603 | 0.669 |

### L2

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.613 | 4.9 | 0.740 | 0.528 |
| pc_cpdag_llm | 0.495 | 5.8 | 0.740 | 0.528 |
| llm_stats_cpdag_greedy | 0.511 | 7.9 | 0.553 | 0.585 |
| llm_stats | 0.254 | 12.1 | 0.439 | 0.393 |
| llm_raw | 0.292 | 12.7 | 0.466 | 0.413 |

### L3

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.697 | 5.5 | 0.779 | 0.629 |
| pc_cpdag_llm | 0.540 | 7.0 | 0.779 | 0.510 |
| llm_stats_cpdag_greedy | 0.460 | 11.2 | 0.493 | 0.390 |
| llm_stats | 0.301 | 20.2 | 0.404 | 0.465 |
| llm_raw | 0.245 | 21.7 | 0.418 | 0.414 |

### L4

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.690 | 6.2 | 0.775 | 0.576 |
| pc_cpdag_llm | 0.622 | 7.0 | 0.781 | 0.587 |
| llm_stats_cpdag_greedy | 0.403 | 12.4 | 0.433 | 0.393 |
| llm_stats | 0.226 | 22.4 | 0.320 | 0.418 |
| llm_raw | 0.281 | 27.0 | 0.372 | 0.493 |

### L5

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.642 | 8.1 | 0.759 | 0.554 |
| pc_cpdag_llm | 0.559 | 9.1 | 0.764 | 0.513 |
| llm_stats_cpdag_greedy | 0.315 | 15.6 | 0.338 | 0.346 |
| llm_stats | 0.158 | 24.0 | 0.254 | 0.202 |
| llm_raw | 0.185 | 28.6 | 0.333 | 0.340 |

### Avg L1–L5

| Method | dir_f1 | SHD | skel_f1 | comp_f1 |
|---|---|---|---|---|
| oracle | 1.000 | 0.0 | 1.000 | 1.000 |
| pc_greedy | 0.689 | 5.3 | 0.780 | 0.612 |
| pc_cpdag_llm | 0.585 | 6.2 | 0.782 | 0.590 |
| llm_stats_cpdag_greedy | 0.454 | 10.3 | 0.509 | 0.479 |
| llm_stats | 0.254 | 16.6 | 0.416 | 0.415 |
| llm_raw | 0.281 | 19.3 | 0.437 | 0.462 |
