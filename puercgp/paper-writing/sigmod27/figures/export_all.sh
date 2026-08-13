#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
paper_dir=$(cd -- "$script_dir/.." && pwd)
build_dir=$(mktemp -d /tmp/graphweft-figure-export.XXXXXX)
trap 'rm -rf -- "$build_dir"' EXIT

figures=(
  01_introduction/core_idea
  02_background/phase_alignment
  03_overview/system_overview
  04_design/planner
  04_design/selector_space
  04_design/state_lifecycle
  05_implementation/operator_mapping
  06_evaluation/planner_speedup
)

for figure_path in "${figures[@]}"; do
  figure_name=${figure_path##*/}
  figure_dir=${figure_path%/*}
  (
    cd "$paper_dir"
    pdflatex \
      -interaction=nonstopmode \
      -halt-on-error \
      -jobname="$figure_name" \
      -output-directory="$build_dir" \
      "\\def\\FigureSource{figures/$figure_path.tex}\\input{figures/export_wrapper.tex}" \
      >/dev/null
  )
  cp -- "$build_dir/$figure_name.pdf" "$script_dir/$figure_dir/$figure_name.pdf"
  printf 'exported %s/%s.pdf\n' "$figure_dir" "$figure_name"
done
