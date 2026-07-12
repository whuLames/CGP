set datafile separator comma
set terminal pngcairo size 1200,760 font "Sans,12"
set grid ytics
set key outside right center
set xlabel "Provisioned SMs"
set ylabel "Kernel latency (ms, log scale)"
set logscale y
set xtics (2,4,8,16,32,48,64,80)

set output "push_sm_scaling.png"
set title "Q=32 shared push: real BFS iteration snapshots"
plot "push_selected.csv" index 0 using 2:9 with linespoints lw 2 pt 7 title "cit step2: 3K V, sharing 1.01", \
     "push_selected.csv" index 1 using 2:9 with linespoints lw 2 pt 7 title "cit step4: 278K V, sharing 1.22", \
     "push_selected.csv" index 2 using 2:9 with linespoints lw 2 pt 7 title "cit step5: 1.22M V, sharing 1.93", \
     "push_selected.csv" index 3 using 2:9 with linespoints lw 2 pt 7 title "cit step8: 3.73M V, sharing 9.61", \
     "push_selected.csv" index 4 using 2:9 with linespoints lw 2 pt 7 title "twitter step3: 7.79M V, sharing 12.55", \
     "push_selected.csv" index 5 using 2:9 with linespoints lw 2 pt 7 title "twitter step6: 21.1M V, sharing 2.75"

set output "pull_sm_scaling.png"
set title "Q=32 fused pull: real BFS states at peak push work"
plot "pull_raw/cit-Patents.csv" using 2:4 with linespoints lw 2 pt 7 title "cit-Patents", \
     "pull_raw/soc-orkut.csv" using 2:4 with linespoints lw 2 pt 7 title "soc-orkut", \
     "pull_raw/soc-twitter.csv" using 2:4 with linespoints lw 2 pt 7 title "soc-twitter", \
     "pull_raw/soc-sinaweibo.csv" using 2:4 with linespoints lw 2 pt 7 title "soc-sinaweibo"
