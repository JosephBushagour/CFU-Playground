# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from nmigen_cfu.util import SimpleElaboratable
from nmigen import unsigned, Signal, Module, Array, Mux

from .stream import Sink, Source, glue_sources
from .sp_mem import MemoryParameters, SinglePortMemory


class InputStore(SimpleElaboratable):
    """Stores input one word at a time and outputs four words at a time.

    Use in this sequence:
    (1) Set num_words
    (2) Set num_words words of input
    (3) Read outputs until done
    (4) Go back to step 1

    Attributes
    ----------

    input: Sink(unsigned(32)), in
      Words written to input are set into the input store.

    next: Signal(), in
      Causes the next four words to be displayed at the output.

    num_words: Sink(unsigned(32)), in
      Number of words in input. When set, resets internal storage.

    output: list of 4 Source(unsigned(32)), out
      The four words of output.

    """
    MAX_WORDS = 2048

    def __init__(self):
        self.input = Sink(unsigned(32))
        self.next = Signal()
        self.num_words = Sink(unsigned(32))
        self.output = [Source(unsigned(32), name=f"output_{i}")
                       for i in range(4)]

    def elab(self, m: Module):
        max_index = Signal(range(self.MAX_WORDS))
        write_index = Signal(range(self.MAX_WORDS))
        read_index = Signal(range(self.MAX_WORDS))
        read_required = Signal()  # Set whenever read_index is updated to initiate a read

        # Create four memories
        memory_params = MemoryParameters(width=32, depth=self.MAX_WORDS // 4)
        memories = Array([SinglePortMemory(params=memory_params)
                         for _ in range(4)])
        for n, memory in enumerate(memories):
            m.submodules[f"mem_{n}"] = memory

        # On read from input sink
        m.d.comb += self.input.ready.eq(1)
        for memory in memories:
            m.d.comb += memory.write_sink.payload.addr.eq(write_index[2:])
            m.d.comb += memory.write_sink.payload.data.eq(self.input.payload),
        with m.If(self.input.is_transferring()):
            m.d.comb += memories[write_index[:2]].write_sink.valid.eq(1),
            with m.If(write_index == max_index - 1):
                # On last word write, wrap back to zero and cause a pre-emptive
                # read
                m.d.sync += write_index.eq(0)
                m.d.sync += read_required.eq(1)
            with m.Else():
                m.d.sync += write_index.eq(write_index + 1)

        # When read_required, push new values down output sinks
        for n, memory in enumerate(memories):
            m.d.comb += glue_sources(memory.read_data_source, self.output[n])
            addr_sink = memory.read_addr_sink
            m.d.sync += addr_sink.payload.eq(read_index[2:])
            with m.If(read_required):
                m.d.sync += addr_sink.valid.eq(1)
                m.d.sync += read_required.eq(0)
            with m.Elif(addr_sink.is_transferring()):
                m.d.sync += addr_sink.valid.eq(0)

        # On "next" signal cause next words to be read by setting read_required
        with m.If(self.next):
            m.d.sync += read_required.eq(1)
            m.d.sync += read_index.eq(Mux(read_index ==
                                      max_index - 4, 0, read_index + 4))

        # Read the value of max_index from num_words sink
        # Reset read and write index too
        m.d.comb += self.num_words.ready.eq(1)
        with m.If(self.num_words.valid):
            m.d.sync += [
                max_index.eq(self.num_words.payload),
                write_index.eq(0),
                read_index.eq(0),
            ]
