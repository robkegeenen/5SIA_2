#!/bin/bash
ITERATIONS=15
LEADER="Clock ticks for"
SEPCHAR="'"

calc() {
  mean="$(echo "($(echo "${@}" | xargs | sed 's/\s\+/+/g'))/${ITERATIONS}" | bc -l)"
  std="$(echo "sqrt(($(echo "${@}" | xargs | sed "s/\(\S\+\)\s*/(\1-${mean})^2+/g")0)/${ITERATIONS})" | bc -l)"
  echo "${mean}${SEPCHAR}${std}"
}

run() {
  declare -A ticks secs
  unset ticks_total secs_total
  for i in $(seq "1" "${ITERATIONS}")
  do
    ttick="0"
    tsec="0"
    while read line
    do
      key="$(echo "${line}" | sed "s/^[^']*'\([^']*\)'.*$/\1/g")"
      tick="$(echo "${line}" | sed "s/^[^:]*:\s*\([^,]*\),.*$/\1/g")"
      ticks["${key}"]="${ticks["${key}"]} ${tick}"
      ttick="$(echo "${ttick} + ${tick}" | bc -l)"
      sec="$(echo "${line}" | sed "s/^[^,]*,\s*\(\S*\)\s.*$/\1/g")"
      secs["${key}"]="${secs["${key}"]} ${sec}"
      tsec="$(echo "${tsec} + ${sec}" | bc -l)"
    done < <(./eeg | grep -P "^${LEADER}")
    ticks_total="${ticks_total} ${ttick}"
    secs_total="${secs_total} ${tsec}"
  done
  echo "name${SEPCHAR}ticks (mean)${SEPCHAR}ticks (std. dev.)${SEPCHAR}seconds (mean)${SEPCHAR}seconds (std. dev.)"
  for key in "${!ticks[@]}"
  do
    echo "${key}${SEPCHAR}$(calc "${ticks["${key}"]}")${SEPCHAR}$(calc "${secs["${key}"]}")"
  done
  echo "total${SEPCHAR}$(calc "${ticks_total}")${SEPCHAR}$(calc "${secs_total}")"
}

run | column -ts"'"
