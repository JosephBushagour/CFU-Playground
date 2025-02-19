// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "perf.h"

#include <stdio.h>

#include "menu.h"

unsigned CFU_start_counts[NUM_PERF_COUNTERS];

void perf_print_human(uint64_t n) {
  if (n > 9999999) {
    printf("%6lluM", (n + 500000) / 1000000);
  } else if (n > 9999) {
    printf("%6lluk", (n + 500) / 1000);
  } else {
    printf("%6llu ", n);
  }
}

void perf_print_value(uint64_t n) {
  perf_print_human(n);
  printf(" (%12llu)", n);
}

// Set each individual perf counter to zero
void perf_reset_all_counters() {
  for (int i = 0; i < NUM_PERF_COUNTERS; ++i) {
    perf_disable_counter(i);
    perf_set_counter(i, 0);
  }
  perf_zero_start_counts();
}

// Prints cycle and enable counts for every perf coutner
void perf_print_all_counters() {
  for (int i = 0; i < NUM_PERF_COUNTERS; ++i) {
    perf_disable_counter(i);
  }
  printf(" Counter |  Total | Starts | Average |     Raw\n");
  printf("---------+--------+--------+---------+--------------\n");
  for (int i = 0; i < NUM_PERF_COUNTERS; ++i) {
    unsigned total = perf_get_counter(i);
    unsigned starts = perf_get_start_count(i);

    printf("  %3d    |", i);
    perf_print_human(total);
    printf(" | %5u  |", starts);
    if (starts) {
      perf_print_human(total / starts);
    } else {
      printf("   n/a ");
    }
    printf("  | %12u\n", total);
  }
}

// Counter Tests
static int counter_num = 0;

static void do_perf_set_0(void) {
  counter_num = 0;
  printf("-curr perf counter %d: %u\n", counter_num,
         perf_get_counter(counter_num));
}

static void do_perf_set_1(void) {
  counter_num = 1;
  printf("-curr perf counter %d: %u\n", counter_num,
         perf_get_counter(counter_num));
}

static void do_perf_enable(void) {
  printf("enable perf counter %d\n", counter_num);
  perf_set_counter_enable(counter_num, 1);
}

static void do_perf_pause(void) {
  printf("pause perf counter %d\n", counter_num);
  perf_set_counter_enable(counter_num, 0);
}

static void do_perf_zero(void) {
  printf("zero perf counter %d and mcycle\n", counter_num);
  perf_set_mcycle(0);
  perf_set_counter_enable(counter_num, 0);
  perf_set_counter(counter_num, 0);
}

static void do_perf_show(void) {
  printf("Counters:\n");
  printf("  0:      %8d\n", perf_get_counter(0));
  printf("  1:      %8d\n", perf_get_counter(1));
  printf("  mcycle: %8d\n", perf_get_mcycle());
}

static struct Menu MENU = {
    "Performance Counter Tests",
    "perf counter",
    {
        MENU_ITEM('0', "switch to perf counter 0 and show value",
                  do_perf_set_0),
        MENU_ITEM('1', "switch to perf counter 1 and show value",
                  do_perf_set_1),
        MENU_ITEM('e', "Enable current perf counter", do_perf_enable),
        MENU_ITEM('p', "Pause current perf counter", do_perf_pause),
        MENU_ITEM('z', "Zero current perf counter and mcycle", do_perf_zero),
        MENU_ITEM('s', "Show perf counters and mcycle values", do_perf_show),
        MENU_END,
    },
};

void perf_test_menu() { menu_run(&MENU); }
