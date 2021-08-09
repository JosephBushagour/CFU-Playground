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

module rcdbpot (
  input logic [31:0] dividend,
  input logic [31:0] exponent,

  output logic [31:0] out
);
  logic [3:0] shift;
  logic signed [31:0] mask, remainder, threshold;
  always_comb begin
    casez (exponent[3:0])
      4'b?111: shift = 9;
      4'b??11: shift = 5;
      4'b???1: shift = 7;
      4'b??1?: shift = 6;
      default: shift = 8;
    endcase
    mask = ~({32{1'b1}} << shift);
    remainder = dividend & mask;
    threshold = (mask >> 1'b1) + dividend[31];
    out = signed'(signed'(dividend) >>> shift)
          + ((remainder > threshold) ? 32'sb1 : 32'sb0);
    out = out[31] ? 32'sd0 : |out[31:8] ? 32'sd255 : out;
    out -= 32'sd128;
  end
endmodule
