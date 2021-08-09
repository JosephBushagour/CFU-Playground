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

module srdhm (
  input logic [31:0] top,
  input logic [31:0] bottom,

  output logic [31:0] out
);
  logic signed [63:0] q_acc;
  assign q_acc = {top, bottom};
  logic signed [31:0] nudge;
  assign nudge = q_acc[63] ? 32'shc0000001 : 32'sh40000000;
  
  assign out = (q_acc + nudge) >> 31;
endmodule
