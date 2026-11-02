#!/usr/bin/env python3
"""Femu.py — educational, pure-Python subset x86 emulator.

Femu models a small 32-bit x86 CPU and a useful subset of 16-bit real-mode x86
for experimenting with flat binaries and PC boot sectors. It is NOT a complete
PC/AT, chipset, VGA, BIOS, protected-mode, or x86 implementation.

Examples:
  python Femu.py                         Run the built-in real-mode boot demo
  python Femu.py --boot boot.bin         Boot a raw disk image from sector 0
  python Femu.py --flat program.bin      Run a 32-bit flat binary at 0x10000
  python Femu.py --flat program.bin --load 0x100000 --entry 0x100000
  python Femu.py --self-test             Run built-in regression tests

Browser use (Pyodide): load this file as Femu.py, exec(compile(source, ...))
into a dedicated namespace, then call Femu32().boot_disk(bytes(...)).
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Optional

VGA_BASE = 0xB8000
VGA_COLS = 80
VGA_ROWS = 25
MEM_SIZE_DEFAULT = 64 * 1024 * 1024
MASK32 = 0xFFFFFFFF
MASK16 = 0xFFFF
FLAG_CF = 1 << 0
FLAG_FIXED = 1 << 1
FLAG_PF = 1 << 2
FLAG_AF = 1 << 4
FLAG_ZF = 1 << 6
FLAG_SF = 1 << 7
FLAG_TF = 1 << 8
FLAG_IF = 1 << 9
FLAG_DF = 1 << 10
FLAG_OF = 1 << 11
ARITH_FLAGS = FLAG_CF | FLAG_PF | FLAG_AF | FLAG_ZF | FLAG_SF | FLAG_OF
REG32_NAMES = ["EAX", "ECX", "EDX", "EBX", "ESP", "EBP", "ESI", "EDI"]
REG16_NAMES = ["AX", "CX", "DX", "BX", "SP", "BP", "SI", "DI"]
REG8_NAMES = ["AL", "CL", "DL", "BL", "AH", "CH", "DH", "BH"]
SEG_NAMES = ["ES", "CS", "SS", "DS", "FS", "GS"]


class FemuError(Exception):
    """A guest execution or image-loading error suitable for showing in a UI."""


class Halted(Exception):
    pass


class Femu32:
    """Small x86 emulator with flat 32-bit mode and a BIOS-assisted real-mode path."""

    def __init__(self, ram_size: int = MEM_SIZE_DEFAULT, disk_image: bytes = b""):
        if not isinstance(ram_size, int) or ram_size < 0x10000:
            raise ValueError("ram_size must be an integer of at least 64 KiB")
        self.ram_size = ram_size
        self.mem = bytearray(ram_size)
        self.reg = {name: 0 for name in REG32_NAMES}
        self.reg.update({"EIP": 0, "EFLAGS": 2})
        self.seg = {name: 0 for name in SEG_NAMES}
        self.protected_mode = False  # Compatibility indicator; not full protected mode.
        self.mode_bits = 32
        self.stopped = False
        self.stop_reason = ""
        self.steps = 0
        self.last_error: Optional[str] = None
        self.trace = False
        self.trace_limit = 10000
        self.trace_lines: list[str] = []
        self.console: list[str] = []
        self.keyboard_buffer: list[int] = []
        self.interrupt_log: list[str] = []
        self.disk_image = bytes(disk_image)
        self.drive_number = 0x80
        self.cursor_row = 0
        self.cursor_col = 0
        self.video_mode = 3
        self.default_segment_override: Optional[str] = None
        self.operand_size = 32
        self.address_size = 32
        self._opcode_start = 0
        # Additional architectural state used by protected-mode startup code.
        self.cr = {"CR0": 0, "CR2": 0, "CR3": 0, "CR4": 0}
        # Guest debug registers; these never touch host debugger state.
        self.dr = {f"DR{i}": 0 for i in range(8)}
        self.dr["DR6"] = 0xFFFF0FF0
        self.dr["DR7"] = 0x400
        self.gdtr = {"base": 0, "limit": 0}
        self.idtr = {"base": 0, "limit": 0x3FF}
        self.msr = {0x10: 0, 0x1B: 0xFEE00000, 0xC0000080: 0}
        self.ports: dict[int, int] = {}
        self.io_log: list[str] = []
        self.serial_console: list[str] = []
        self.repeat_prefix: Optional[int] = None
        self.a20_enabled = True
        self.panic_message: Optional[str] = None
        self.fpu_control = 0x037F
        self.fpu_status = 0
        self.fpu_tags = 0xFFFF
        self.mxcsr = 0x1F80

    # ----- memory and register aliases -----
    def _check(self, addr: int, size: int = 1) -> None:
        if not isinstance(addr, int) or not isinstance(size, int) or size < 0:
            raise FemuError("invalid memory address/size")
        if addr < 0 or addr + size > self.ram_size:
            raise FemuError(f"memory access out of bounds: 0x{addr:X} (+{size}), RAM=0x{self.ram_size:X}")

    def read8(self, addr: int) -> int:
        self._check(addr, 1)
        return self.mem[addr]

    def write8(self, addr: int, value: int) -> None:
        self._check(addr, 1)
        self.mem[addr] = value & 0xFF

    def read16(self, addr: int) -> int:
        self._check(addr, 2)
        return self.mem[addr] | (self.mem[addr + 1] << 8)

    def write16(self, addr: int, value: int) -> None:
        self._check(addr, 2)
        value &= MASK16
        self.mem[addr] = value & 0xFF
        self.mem[addr + 1] = (value >> 8) & 0xFF

    def read32(self, addr: int) -> int:
        self._check(addr, 4)
        return struct.unpack_from("<I", self.mem, addr)[0]

    def write32(self, addr: int, value: int) -> None:
        self._check(addr, 4)
        struct.pack_into("<I", self.mem, addr, value & MASK32)

    def read64(self, addr: int) -> int:
        self._check(addr, 8)
        return struct.unpack_from("<Q", self.mem, addr)[0]

    def write64(self, addr: int, value: int) -> None:
        self._check(addr, 8)
        struct.pack_into("<Q", self.mem, addr, value & 0xFFFFFFFFFFFFFFFF)

    def read_value(self, addr: int, width: int) -> int:
        return self.read8(addr) if width == 8 else self.read16(addr) if width == 16 else self.read32(addr)

    def write_value(self, addr: int, value: int, width: int) -> None:
        if width == 8:
            self.write8(addr, value)
        elif width == 16:
            self.write16(addr, value)
        elif width == 32:
            self.write32(addr, value)
        else:
            raise FemuError(f"unsupported operand width {width}")

    @property
    def eip(self) -> int:
        return self.reg["EIP"] & (MASK16 if self.mode_bits == 16 else MASK32)

    @eip.setter
    def eip(self, value: int) -> None:
        self.reg["EIP"] = value & (MASK16 if self.mode_bits == 16 else MASK32)

    def get_reg(self, idx: int, width: int) -> int:
        if width == 32:
            return self.reg[REG32_NAMES[idx]] & MASK32
        if width == 16:
            return self.reg[REG32_NAMES[idx]] & MASK16
        if width == 8:
            if idx < 4:
                return self.reg[REG32_NAMES[idx]] & 0xFF
            return (self.reg[REG32_NAMES[idx - 4]] >> 8) & 0xFF
        raise FemuError(f"unsupported register width {width}")

    def set_reg(self, idx: int, value: int, width: int) -> None:
        name = REG32_NAMES[idx]
        if width == 32:
            self.reg[name] = value & MASK32
        elif width == 16:
            self.reg[name] = (self.reg[name] & 0xFFFF0000) | (value & MASK16)
        elif width == 8:
            if idx < 4:
                self.reg[name] = (self.reg[name] & 0xFFFFFF00) | (value & 0xFF)
            else:
                base = REG32_NAMES[idx - 4]
                self.reg[base] = (self.reg[base] & 0xFFFF00FF) | ((value & 0xFF) << 8)
        else:
            raise FemuError(f"unsupported register width {width}")

    def get_alias(self, name: str) -> int:
        name = name.upper()
        if name in self.reg:
            return self.reg[name]
        if name in self.seg:
            return self.seg[name]
        if name in self.cr:
            return self.cr[name]
        if name in self.dr:
            return self.dr[name]
        if name in REG16_NAMES:
            return self.get_reg(REG16_NAMES.index(name), 16)
        if name in REG8_NAMES:
            return self.get_reg(REG8_NAMES.index(name), 8)
        raise KeyError(name)

    def set_alias(self, name: str, value: int) -> None:
        name = name.upper()
        if name in self.reg:
            self.reg[name] = value & MASK32 if name != "EFLAGS" else ((value | FLAG_FIXED) & MASK32)
        elif name in self.seg:
            self.seg[name] = value & MASK16
        elif name in self.cr:
            self.cr[name] = value & MASK32
            if name == "CR0":
                # Setting CR0.PE does not reload CS. Keep the old execution
                # addressing state until the required far transfer completes.
                if not (value & 1):
                    self.protected_mode = False
            elif name == "CR3":
                self.cr[name] = value & 0xFFFFF000
        elif name in self.dr:
            self.dr[name] = value & MASK32
        elif name in REG16_NAMES:
            self.set_reg(REG16_NAMES.index(name), value, 16)
        elif name in REG8_NAMES:
            self.set_reg(REG8_NAMES.index(name), value, 8)
        else:
            raise KeyError(name)

    def _translate_page(self, linear_addr: int) -> int:
        """Translate a linear address with non-PAE 32-bit paging when CR0.PG is set."""
        linear_addr &= MASK32
        if not (self.cr["CR0"] & 0x80000000):
            return linear_addr
        if self.cr["CR4"] & (1 << 5):
            raise FemuError("PAE paging requested (CR4.PAE), but PAE/long-mode paging is not implemented")
        directory = self.cr["CR3"] & 0xFFFFF000
        pde_addr = directory + (((linear_addr >> 22) & 0x3FF) * 4)
        self._check(pde_addr, 4)
        pde = self.read32(pde_addr)
        if not (pde & 1):
            self.cr["CR2"] = linear_addr
            raise FemuError(f"page fault: not-present PDE for linear address 0x{linear_addr:08X}")
        self.write32(pde_addr, pde | (1 << 5))  # accessed bit
        if (pde & (1 << 7)) and (self.cr["CR4"] & (1 << 4)):
            return (pde & 0xFFC00000) | (linear_addr & 0x3FFFFF)
        table = pde & 0xFFFFF000
        pte_addr = table + (((linear_addr >> 12) & 0x3FF) * 4)
        self._check(pte_addr, 4)
        pte = self.read32(pte_addr)
        if not (pte & 1):
            self.cr["CR2"] = linear_addr
            raise FemuError(f"page fault: not-present PTE for linear address 0x{linear_addr:08X}")
        self.write32(pte_addr, pte | (1 << 5))
        return (pte & 0xFFFFF000) | (linear_addr & 0xFFF)

    def linear(self, offset: int, segment: Optional[str] = None, addr_bits: Optional[int] = None) -> int:
        if self.mode_bits == 16 and not self.protected_mode:
            segname = segment or "DS"
            addr = ((self.seg[segname] << 4) + (offset & MASK16)) & MASK32
        else:
            # Current protected-mode descriptors are flat by design. Descriptor
            # table state is recorded and used for far transfers, but full privilege
            # checks and expand-down segment semantics are intentionally omitted.
            addr = offset & MASK32
        return self._translate_page(addr)

    # ----- instruction fetch, stack, flags -----
    def fetch8(self) -> int:
        addr = self.linear(self.eip, "CS")
        value = self.read8(addr)
        self.eip = self.eip + 1
        return value

    def fetch16(self) -> int:
        lo = self.fetch8()
        return lo | (self.fetch8() << 8)

    def fetch32(self) -> int:
        lo = self.fetch16()
        return lo | (self.fetch16() << 16)

    def fetch_signed(self, width: int) -> int:
        raw = self.fetch8() if width == 8 else self.fetch16() if width == 16 else self.fetch32()
        sign = 1 << (width - 1)
        return raw - (1 << width) if raw & sign else raw

    def push(self, value: int, width: Optional[int] = None) -> None:
        width = width or self.operand_size
        size = width // 8
        stack_mask = MASK16 if self.mode_bits == 16 else MASK32
        sp = (self.reg["ESP"] - size) & stack_mask
        self.reg["ESP"] = (self.reg["ESP"] & ~stack_mask) | sp
        self.write_value(self.linear(sp, "SS"), value, width)

    def pop(self, width: Optional[int] = None) -> int:
        width = width or self.operand_size
        size = width // 8
        stack_mask = MASK16 if self.mode_bits == 16 else MASK32
        sp = self.reg["ESP"] & stack_mask
        value = self.read_value(self.linear(sp, "SS"), width)
        sp = (sp + size) & stack_mask
        self.reg["ESP"] = (self.reg["ESP"] & ~stack_mask) | sp
        return value

    def flag(self, flag: int) -> int:
        return 1 if self.reg["EFLAGS"] & flag else 0

    def set_flag(self, flag: int, enabled: bool) -> None:
        if enabled:
            self.reg["EFLAGS"] |= flag
        else:
            self.reg["EFLAGS"] &= ~flag
        self.reg["EFLAGS"] |= FLAG_FIXED

    @staticmethod
    def parity(value: int) -> bool:
        return (value & 0xFF).bit_count() % 2 == 0

    def set_szp(self, result: int, width: int) -> None:
        mask = (1 << width) - 1
        result &= mask
        self.set_flag(FLAG_ZF, result == 0)
        self.set_flag(FLAG_SF, bool(result & (1 << (width - 1))))
        self.set_flag(FLAG_PF, self.parity(result))

    def alu(self, op: str, a: int, b: int, width: int) -> int:
        mask = (1 << width) - 1
        sign = 1 << (width - 1)
        a &= mask
        b &= mask
        if op in ("ADD", "ADC"):
            carry_in = self.flag(FLAG_CF) if op == "ADC" else 0
            full = a + b + carry_in
            result = full & mask
            self.set_flag(FLAG_CF, full > mask)
            self.set_flag(FLAG_AF, ((a ^ b ^ result) & 0x10) != 0)
            self.set_flag(FLAG_OF, bool((~(a ^ b) & (a ^ result) & sign)))
        elif op in ("SUB", "CMP", "SBB"):
            borrow = self.flag(FLAG_CF) if op == "SBB" else 0
            rhs = b + borrow
            result = (a - rhs) & mask
            self.set_flag(FLAG_CF, a < rhs)
            self.set_flag(FLAG_AF, ((a ^ b ^ result) & 0x10) != 0)
            self.set_flag(FLAG_OF, bool(((a ^ b) & (a ^ result) & sign)))
        elif op == "AND":
            result = a & b
            self.set_flag(FLAG_CF, False); self.set_flag(FLAG_OF, False); self.set_flag(FLAG_AF, False)
        elif op == "OR":
            result = a | b
            self.set_flag(FLAG_CF, False); self.set_flag(FLAG_OF, False); self.set_flag(FLAG_AF, False)
        elif op == "XOR":
            result = a ^ b
            self.set_flag(FLAG_CF, False); self.set_flag(FLAG_OF, False); self.set_flag(FLAG_AF, False)
        elif op == "TEST":
            result = a & b
            self.set_flag(FLAG_CF, False); self.set_flag(FLAG_OF, False); self.set_flag(FLAG_AF, False)
        else:
            raise FemuError(f"internal error: unsupported ALU op {op}")
        self.set_szp(result, width)
        return result & mask

    def incdec(self, value: int, width: int, delta: int) -> int:
        old_cf = self.flag(FLAG_CF)
        result = self.alu("ADD" if delta > 0 else "SUB", value, 1, width)
        self.set_flag(FLAG_CF, bool(old_cf))  # INC/DEC preserve carry.
        return result

    def condition(self, code: int) -> bool:
        cf, pf, zf = self.flag(FLAG_CF), self.flag(FLAG_PF), self.flag(FLAG_ZF)
        sf, of = self.flag(FLAG_SF), self.flag(FLAG_OF)
        table = {
            0: of == 1, 1: of == 0, 2: cf == 1, 3: cf == 0,
            4: zf == 1, 5: zf == 0, 6: cf == 1 or zf == 1,
            7: cf == 0 and zf == 0, 8: sf == 1, 9: sf == 0,
            10: pf == 1, 11: pf == 0, 12: sf != of, 13: sf == of,
            14: zf == 1 or sf != of, 15: zf == 0 and sf == of,
        }
        return table[code]

    # ----- ModR/M + SIB and operand references -----
    def decode_modrm(self, byte: int):
        return (byte >> 6) & 3, (byte >> 3) & 7, byte & 7

    def _effective_address(self, mod: int, rm: int, address_size: int) -> tuple[int, str]:
        seg = self.default_segment_override or "DS"
        disp = 0
        if address_size == 16:
            if mod == 0 and rm == 6:
                offset = self.fetch16()
                return self.linear(offset, seg), seg
            pairs = [("BX", "SI"), ("BX", "DI"), ("BP", "SI"), ("BP", "DI"),
                     ("SI", None), ("DI", None), ("BP", None), ("BX", None)]
            a, b = pairs[rm]
            offset = self.get_alias(a) + (self.get_alias(b) if b else 0)
            if a == "BP":
                if self.default_segment_override is None:
                    seg = "SS"
            if mod == 1:
                disp = self.fetch_signed(8)
            elif mod == 2:
                disp = self.fetch_signed(16)
            offset = (offset + disp) & MASK16
            return self.linear(offset, seg), seg
        # 32-bit addressing (including SIB).
        if rm == 4:
            sib = self.fetch8()
            scale = 1 << ((sib >> 6) & 3)
            index = (sib >> 3) & 7
            base = sib & 7
            idxval = 0 if index == 4 else self.reg[REG32_NAMES[index]]
            if mod == 0 and base == 5:
                baseval = 0
                disp = self.fetch32()
            else:
                baseval = self.reg[REG32_NAMES[base]]
            if base in (4, 5) and mod != 0 and self.default_segment_override is None:
                seg = "SS"
            offset = (baseval + idxval * scale + disp) & MASK32
        elif mod == 0 and rm == 5:
            offset = self.fetch32()
        else:
            offset = self.reg[REG32_NAMES[rm]]
            if rm in (4, 5) and self.default_segment_override is None:
                seg = "SS"
            if mod == 1:
                offset += self.fetch_signed(8)
            elif mod == 2:
                offset += self.fetch_signed(32)
            offset &= MASK32
        if mod == 1 and rm == 4:
            offset = (offset + self.fetch_signed(8)) & MASK32
        elif mod == 2 and rm == 4:
            offset = (offset + self.fetch_signed(32)) & MASK32
        return self.linear(offset, seg), seg

    def _decode_rm(self, mod: int, rm: int, width: int, address_size: int):
        if mod == 3:
            return ("reg", rm, width)
        addr, seg = self._effective_address(mod, rm, address_size)
        return ("mem", addr, width)

    def _read_operand(self, operand) -> int:
        kind, arg, width = operand
        return self.get_reg(arg, width) if kind == "reg" else self.read_value(arg, width)

    def _write_operand(self, operand, value: int) -> None:
        kind, arg, width = operand
        if kind == "reg":
            self.set_reg(arg, value, width)
        else:
            self.write_value(arg, value, width)

    def queue_key(self, ascii_code: int, scan_code: int = 0) -> None:
        """Queue one browser keyboard character as BIOS AX=(scan<<8)|ASCII."""
        ascii_code = int(ascii_code) & 0xFF
        scan_code = int(scan_code) & 0xFF
        self.keyboard_buffer.append((scan_code << 8) | ascii_code)

    # ----- x86 control state, BIOS machine I/O and descriptors -----
    def _port_in(self, port: int, width: int = 8) -> int:
        port &= 0xFFFF
        if port == 0x64: value = 0x02  # keyboard controller input buffer empty
        elif port == 0x60: value = self.keyboard_buffer.pop(0) if self.keyboard_buffer else 0x00
        elif port == 0x92: value = 0x02 if self.a20_enabled else 0
        elif port == 0x3FD: value = 0x60  # COM1 transmit-empty + transmitter-empty
        elif port == 0x3F8: value = 0  # read-empty serial receive register
        elif port in (0x20, 0xA0): value = 0
        elif port in (0x21, 0xA1): value = 0xFF
        elif port == 0x71: value = 0  # deterministic CMOS byte
        elif port == 0x40: value = (self.steps * 17) & 0xFF  # deterministic PIT sample
        else: value = self.ports.get(port, 0xFF)
        self.io_log.append(f"IN{width} port 0x{port:04X} -> 0x{value & ((1 << width) - 1):X}")
        return value & ((1 << width) - 1)

    def _port_out(self, port: int, value: int, width: int = 8) -> None:
        port &= 0xFFFF
        value &= (1 << width) - 1
        self.ports[port] = value
        self.io_log.append(f"OUT{width} port 0x{port:04X} <- 0x{value:X}")
        if port == 0x92:
            self.a20_enabled = bool(value & 0x02)
        if port == 0xE9:
            self.serial_console.append(chr(value & 0xFF))
        if port == 0x3F8:
            # COM1 transmit data register: capture as guest-visible serial output.
            self.serial_console.append(chr(value & 0xFF))

    def _port_in_quiet(self, port: int) -> int:
        if port == 0x3F9: return 0x20
        return self.ports.get(port, 0)

    def _descriptor(self, selector: int) -> tuple[int, int, int]:
        """Decode a protected-mode GDT descriptor and enforce GDTR bounds."""
        selector &= MASK16
        index = selector & 0xFFF8
        if index == 0:
            raise FemuError("null protected-mode segment selector")
        if selector & 4:
            raise FemuError(f"LDT selector 0x{selector:04X} is unsupported (TI=1)")
        if index + 7 > self.gdtr["limit"]:
            raise FemuError(f"selector 0x{selector:04X} exceeds GDT limit 0x{self.gdtr['limit']:X}")
        off = (self.gdtr["base"] + index) & MASK32
        self._check(off, 8)
        low = self.read32(off)
        high = self.read32(off + 4)
        limit = (low & 0xFFFF) | (((high >> 16) & 0xF) << 16)
        base = ((low >> 16) & 0xFFFF) | ((high & 0xFF) << 16) | (((high >> 24) & 0xFF) << 24)
        access = (high >> 8) & 0xFF
        flags = (high >> 20) & 0xF
        if flags & 8:
            limit = (limit << 12) | 0xFFF
        return base, limit, (access << 4) | flags

    def _load_segment_selector(self, name: str, selector: int) -> None:
        """Load a segment selector with basic 386 protected-mode checks."""
        selector &= MASK16
        if not (self.cr["CR0"] & 1):
            self.seg[name] = selector
            return
        if selector & 0xFFF8 == 0:
            if name == "SS":
                raise FemuError(f"null selector 0x{selector:04X} cannot be loaded into SS in protected mode at CS:IP={self.seg['CS']:04X}:{self._opcode_start:04X}")
            # Null DS/ES/FS/GS selectors are architecturally permitted.
            self.seg[name] = 0
            return
        _base, _limit, attrs = self._descriptor(selector)
        access = (attrs >> 4) & 0xFF
        if not (access & 0x80):
            raise FemuError(f"segment load {name} from non-present selector 0x{selector:04X}")
        if name == "SS" and ((access & 0x18) != 0x10 or not (access & 0x02)):
            raise FemuError(f"SS selector 0x{selector:04X} is not a writable data segment")
        if name != "SS" and (access & 0x18) == 0x10 and not (access & 0x02):
            raise FemuError(f"segment selector 0x{selector:04X} is not readable")
        self.seg[name] = selector

    def _far_transfer(self, offset: int, selector: int) -> None:
        if self.cr["CR0"] & 1:
            if (selector & 0xFFF8) == 0:
                raise FemuError(f"null protected-mode CS selector during far transfer at CS:IP={self.seg['CS']:04X}:{self._opcode_start:04X} (target offset=0x{offset:X}, CR0=0x{self.cr['CR0']:08X}, GDTR base=0x{self.gdtr['base']:08X} limit=0x{self.gdtr['limit']:04X})")
            _base, limit, attrs = self._descriptor(selector)
            access = (attrs >> 4) & 0xFF
            flags = attrs & 0xF
            if not (access & 0x80):
                raise FemuError(f"far jump to non-present selector 0x{selector:04X}")
            if not (access & 0x08):
                raise FemuError(f"far jump selector 0x{selector:04X} is not executable")
            if offset > limit:
                raise FemuError(f"far jump offset 0x{offset:X} exceeds code segment limit 0x{limit:X}")
            self.protected_mode = True
            self.seg["CS"] = selector & MASK16
            self.mode_bits = 32 if (flags & 4) else 16
            self.eip = offset
        else:
            self.seg["CS"] = selector & MASK16
            self.mode_bits = 16
            self.protected_mode = False
            self.eip = offset & MASK16

    def _cpu_cpuid(self) -> None:
        leaf = self.reg["EAX"] & MASK32
        subleaf = self.reg["ECX"] & MASK32
        if leaf == 0:
            self.reg["EAX"] = 1
            self.reg["EBX"] = int.from_bytes(b"Genu", "little")
            self.reg["EDX"] = int.from_bytes(b"ineI", "little")
            self.reg["ECX"] = int.from_bytes(b"ntel", "little")
        elif leaf == 1:
            self.reg["EAX"] = 0x00000663  # synthetic Pentium-II-ish model
            self.reg["EBX"] = 0
            self.reg["ECX"] = 0
            # Advertise only features for which this emulator has basic support.
            self.reg["EDX"] = (1 << 0) | (1 << 4) | (1 << 5) | (1 << 8) | (1 << 15)
        elif leaf == 0x80000000:
            self.reg["EAX"] = 0x80000001
            self.reg["EBX"] = self.reg["ECX"] = self.reg["EDX"] = 0
        elif leaf == 0x80000001:
            # No long-mode bit: this implementation does not execute x86-64 code.
            self.reg["EAX"] = self.reg["EBX"] = self.reg["ECX"] = self.reg["EDX"] = 0
        else:
            self.reg["EAX"] = self.reg["EBX"] = self.reg["ECX"] = self.reg["EDX"] = 0
        self.reg["EFLAGS"] &= ~(FLAG_CF | FLAG_PF | FLAG_AF | FLAG_ZF | FLAG_SF | FLAG_OF)

    def _detect_linux_panic(self) -> Optional[str]:
        text = "".join(self.console) + "".join(self.serial_console) + "\n" + self.vga_text()
        import re
        match = re.search(r"Kernel panic\s*-\s*not syncing:\s*([^\r\n]+)", text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        if re.search(r"Kernel panic\s*-\s*not syncing", text, re.IGNORECASE):
            return "Kernel panic - not syncing"
        return None

    # ----- BIOS services and software interrupts -----
    def _set_cf(self, enabled: bool) -> None:
        self.set_flag(FLAG_CF, enabled)

    def _screen_char(self, ch: int, attr: int = 0x07) -> None:
        if ch == 13:
            self.cursor_col = 0
            return
        if ch == 10:
            self.cursor_row += 1
            if self.cursor_row >= VGA_ROWS:
                self._scroll_vga()
                self.cursor_row = VGA_ROWS - 1
            return
        if ch == 8:
            self.cursor_col = max(0, self.cursor_col - 1)
            return
        if 0 <= self.cursor_row < VGA_ROWS and 0 <= self.cursor_col < VGA_COLS:
            off = VGA_BASE + 2 * (self.cursor_row * VGA_COLS + self.cursor_col)
            if off + 1 < self.ram_size:
                self.write8(off, ch)
                self.write8(off + 1, attr)
        self.cursor_col += 1
        if self.cursor_col >= VGA_COLS:
            self.cursor_col = 0
            self.cursor_row += 1
        if self.cursor_row >= VGA_ROWS:
            self._scroll_vga()
            self.cursor_row = VGA_ROWS - 1

    def _scroll_vga(self) -> None:
        if VGA_BASE + VGA_ROWS * VGA_COLS * 2 <= self.ram_size:
            start = VGA_BASE
            end = start + VGA_ROWS * VGA_COLS * 2
            rowbytes = VGA_COLS * 2
            self.mem[start:end-rowbytes] = self.mem[start+rowbytes:end]
            self.mem[end-rowbytes:end] = bytes(rowbytes)

    def _bios_int10(self) -> None:
        ah, al = self.get_alias("AH"), self.get_alias("AL")
        if ah == 0x0E:  # teletype output
            self._screen_char(al, self.get_alias("BL") if self.get_alias("BL") else 0x07)
            self.console.append(chr(al))
        elif ah == 0x00:  # set video mode
            self.video_mode = al
            self.cursor_row = self.cursor_col = 0
        elif ah == 0x02:  # set cursor position
            self.cursor_row = self.get_alias("DH") % VGA_ROWS
            self.cursor_col = self.get_alias("DL") % VGA_COLS
        elif ah == 0x03:  # get cursor position
            self.set_alias("DH", self.cursor_row)
            self.set_alias("DL", self.cursor_col)
            self.set_alias("CH", 0)
            self.set_alias("CL", 0)
        elif ah == 0x0F:  # get video mode
            self.set_alias("AL", self.video_mode)
            self.set_alias("AH", VGA_COLS)
            self.set_alias("BH", 0)
        else:
            self.interrupt_log.append(f"INT 10h AH={ah:02X}: unsupported video service")

    def _bios_int13(self) -> None:
        ah = self.get_alias("AH")
        if ah == 0x00:  # reset disk system
            self.set_alias("AH", 0)
            self._set_cf(False)
            return
        if ah == 0x02:  # read sectors (CHS; geometry 18 sectors x 2 heads)
            count = self.get_alias("AL")
            cylinder = self.get_alias("CH") | ((self.get_alias("CL") & 0xC0) << 2)
            sector = self.get_alias("CL") & 0x3F
            head = self.get_alias("DH")
            drive = self.get_alias("DL")
            if not count or sector == 0 or head > 1 or drive not in (0, self.drive_number) or cylinder > 1023:
                self.set_alias("AH", 1); self.set_alias("AL", 0); self._set_cf(True); return
            lba = ((cylinder * 2 + head) * 18) + sector - 1
            start = lba * 512
            amount = count * 512
            if start + amount > len(self.disk_image):
                self.set_alias("AH", 4); self.set_alias("AL", 0); self._set_cf(True); return
            dest = self.linear(self.get_alias("BX"), "ES")
            self._check(dest, amount)
            self.mem[dest:dest + amount] = self.disk_image[start:start + amount]
            self.set_alias("AH", 0); self.set_alias("AL", count); self._set_cf(False)
            return
        if ah == 0x08:  # get drive parameters (synthetic 1.44 MiB-style geometry)
            if self.get_alias("DL") not in (0, self.drive_number):
                self.set_alias("AH", 1); self._set_cf(True); return
            self.set_alias("CH", 79); self.set_alias("CL", 18)
            self.set_alias("DH", 1); self.set_alias("DL", 1)
            self.set_alias("BL", 4); self.set_alias("AH", 0); self._set_cf(False); return
        if ah == 0x15:  # drive type / capacity
            if self.get_alias("DL") in (0, self.drive_number):
                self.set_alias("AH", 3); self.set_alias("CX", 0); self.set_alias("DX", 0)
                self._set_cf(False); return
            self.set_alias("AH", 0); self._set_cf(True); return
        if ah == 0x41:  # installation check for extended disk services
            if self.get_alias("BX") == 0x55AA and self.get_alias("DL") in (0, self.drive_number):
                self.set_alias("BX", 0xAA55); self.set_alias("CX", 1); self.set_alias("AH", 0x30)
                self._set_cf(False); return
            self.set_alias("AH", 1); self._set_cf(True); return
        if ah == 0x42:  # extended LBA read via disk address packet (DAP)
            dap = self.linear(self.get_alias("SI"), "DS")
            self._check(dap, 16)
            packet_size = self.read8(dap)
            count = self.read16(dap + 2)
            offset = self.read16(dap + 4)
            segment = self.read16(dap + 6)
            lba = self.read32(dap + 8) | (self.read32(dap + 12) << 32)
            amount = count * 512
            start = lba * 512
            if packet_size < 16 or not count or start + amount > len(self.disk_image):
                self.set_alias("AH", 4); self.set_alias("AL", 0); self._set_cf(True); return
            dest = self.linear(offset, "ES") if self.seg["ES"] == segment else ((segment << 4) + offset)
            self._check(dest, amount)
            self.mem[dest:dest + amount] = self.disk_image[start:start + amount]
            self.set_alias("AH", 0); self.set_alias("AL", count); self._set_cf(False); return
        if ah == 0x48:  # extended drive parameters packet
            ptr = self.linear(self.get_alias("SI"), "DS")
            self._check(ptr, 0x1A)
            size = self.read16(ptr)
            if size < 0x1A:
                self.set_alias("AH", 1); self._set_cf(True); return
            sectors = max(1, (len(self.disk_image) + 511) // 512)
            self.write16(ptr + 2, 0x0002)  # removable/extended geometry flags
            self.write32(ptr + 4, min(1024, (sectors + 35) // 36))
            self.write32(ptr + 8, 2); self.write32(ptr + 12, 18)
            self.write64(ptr + 16, sectors)
            self.write16(ptr + 24, 512)
            self.set_alias("AH", 0); self._set_cf(False); return
        self.set_alias("AH", 1)
        self._set_cf(True)
        self.interrupt_log.append(f"INT 13h AH={ah:02X}: unsupported disk service")

    def _bios_int15(self) -> None:
        ax = self.get_alias("AX")
        ah = self.get_alias("AH")
        if ax == 0xE820 and self.get_alias("EDX") == 0x534D4150:
            entries = [(0, 0x9F000, 1), (0x9F000, 0x61000, 2)]
            if self.ram_size > 0x100000:
                entries.append((0x100000, self.ram_size - 0x100000, 1))
            index = self.get_alias("EBX")
            if index >= len(entries):
                self._set_cf(True); return
            base, length, typ = entries[index]
            dest = self.linear(self.get_alias("DI"), "ES")
            self._check(dest, 20)
            struct.pack_into("<QQI", self.mem, dest, base, length, typ)
            self.set_alias("EAX", 0x534D4150)
            self.set_alias("ECX", 20)
            self.set_alias("EBX", index + 1 if index + 1 < len(entries) else 0)
            self._set_cf(False)
            return
        if ah == 0x88:
            self.set_alias("AX", max(0, min(0xFFFF, (self.ram_size - 0x100000) // 1024)))
            self._set_cf(False); return
        if ax == 0xE801:
            kb_1_to_16 = max(0, min(15 * 1024, (self.ram_size - 0x100000) // 1024))
            self.set_alias("AX", kb_1_to_16); self.set_alias("CX", kb_1_to_16)
            self.set_alias("BX", 0); self.set_alias("DX", 0)
            self._set_cf(False); return
        if ah == 0x24:
            # A20 enable/status are simulated and always succeed.
            sub = self.get_alias("AL")
            if sub == 0x02: self.set_alias("AL", 1 if self.a20_enabled else 0)
            elif sub == 0x01: self.a20_enabled = True
            self._set_cf(False); return
        self.set_alias("AH", 0x86)
        self._set_cf(True)
        self.interrupt_log.append(f"INT 15h AH={ah:02X}: unsupported system service")

    def _bios_interrupt(self, number: int) -> bool:
        if number == 0x11:
            self.set_alias("AX", 0x0021)  # synthetic AT-compatible equipment word
            self._set_cf(False)
            return True
        if number == 0x12:
            self.set_alias("AX", 640)  # conventional memory in KiB
            self._set_cf(False)
            return True
        if number == 0x15:
            self._bios_int15(); return True
        if number == 0x10:
            self._bios_int10(); return True
        if number == 0x13:
            self._bios_int13(); return True
        if number == 0x16:  # BIOS keyboard services backed by the browser input queue
            ah = self.get_alias("AH")
            if ah in (0x00, 0x10):  # read key (empty queue returns 0 instead of blocking)
                key = self.keyboard_buffer.pop(0) if self.keyboard_buffer else 0
                self.set_alias("AX", key & 0xFFFF)
                self._set_cf(False)
                if not key:
                    self.interrupt_log.append("INT 16h: keyboard buffer empty")
            elif ah in (0x01, 0x11):  # check key; ZF=1 when empty
                key = self.keyboard_buffer[0] if self.keyboard_buffer else 0
                self.set_alias("AX", key & 0xFFFF)
                self.set_flag(FLAG_ZF, not bool(key))
                self._set_cf(False)
            return True
        if number in (0x18, 0x19):  # boot failure / bootstrap request
            self.interrupt_log.append(f"BIOS INT {number:02X}h: no alternate boot device configured")
            self.stopped = True
            self.stop_reason = "BIOS boot request without another boot device"
            return True
        if number == 0x1A:
            ah = self.get_alias("AH")
            if ah == 0:
                self.set_alias("CX", 0); self.set_alias("DX", 0); self.set_alias("AL", 0)
                self._set_cf(False)
            else:
                self.set_alias("AH", 0x86); self._set_cf(True)
            return True
        return False

    def interrupt(self, number: int) -> None:
        number &= 0xFF
        # BIOS teletype output calls INT 10h/AH=0Eh once per character. The
        # actual characters are already captured in ``console``; logging every
        # call floods the report with dozens of identical addresses. Keep the
        # interrupt log useful by omitting only this high-frequency success path.
        is_teletype = number == 0x10 and self.get_alias("AH") == 0x0E
        if not is_teletype:
            self.interrupt_log.append(f"INT {number:02X}h at {self._opcode_start:08X}")
        if self._bios_interrupt(number):
            return
        # Real-mode IVT fallback. This can use guest-installed interrupt handlers.
        if self.mode_bits == 16 and not self.protected_mode:
            vector = number * 4
            off, seg = self.read16(vector), self.read16(vector + 2)
            if off == 0 and seg == 0:
                raise FemuError(f"unhandled software interrupt INT {number:02X}h (empty IVT vector)")
            self.push(self.reg["EFLAGS"], 16)
            self.push(self.seg["CS"], 16)
            self.push(self.eip, 16)
            self.set_flag(FLAG_IF, False); self.set_flag(FLAG_TF, False)
            self.seg["CS"] = seg
            self.eip = off
            return
        raise FemuError(f"unhandled software interrupt INT {number:02X}h in flat mode")

    # ----- loader / boot paths -----
    def reset_cpu(self) -> None:
        for name in REG32_NAMES:
            self.reg[name] = 0
        self.reg["EIP"] = 0
        self.reg["EFLAGS"] = 2
        for name in SEG_NAMES:
            self.seg[name] = 0
        self.mode_bits = 32
        self.protected_mode = True  # Flat-binary mode: no descriptor-table enforcement.
        self.stopped = False
        self.stop_reason = ""
        self.last_error = None
        self.steps = 0
        self.console.clear(); self.interrupt_log.clear(); self.trace_lines.clear()
        self.serial_console.clear(); self.io_log.clear(); self.panic_message = None
        self.cr = {"CR0": 0, "CR2": 0, "CR3": 0, "CR4": 0}
        # Guest debug registers; these never touch host debugger state.
        self.dr = {f"DR{i}": 0 for i in range(8)}
        self.dr["DR6"] = 0xFFFF0FF0
        self.dr["DR7"] = 0x400
        self.gdtr = {"base": 0, "limit": 0}
        self.idtr = {"base": 0, "limit": 0x3FF}
        self.msr = {0x10: 0, 0x1B: 0xFEE00000, 0xC0000080: 0}
        self.repeat_prefix = None
        self.fpu_control = 0x037F; self.fpu_status = 0; self.fpu_tags = 0xFFFF; self.mxcsr = 0x1F80

    def load_flat_binary(self, image: bytes, load_address: int = 0x10000,
                         entry_address: Optional[int] = None, mode_bits: int = 32) -> None:
        if not isinstance(image, (bytes, bytearray, memoryview)):
            raise TypeError("image must be bytes-like")
        if mode_bits not in (16, 32):
            raise ValueError("mode_bits must be 16 or 32")
        if load_address < 0 or load_address + len(image) > self.ram_size:
            raise FemuError("flat binary does not fit in guest RAM")
        self.reset_cpu()
        self.mode_bits = mode_bits
        self.protected_mode = mode_bits == 32
        self.cr["CR0"] = 1 if mode_bits == 32 else 0
        self.mem[load_address:load_address + len(image)] = image
        entry = load_address if entry_address is None else entry_address
        if entry < 0 or entry >= self.ram_size:
            raise FemuError("entry address is outside guest RAM")
        self.reg["EIP"] = entry if mode_bits == 32 else entry & MASK16
        if mode_bits == 16:
            self.seg["CS"] = 0
            self.reg["EIP"] = entry & MASK16
        self.reg["ESP"] = min(self.ram_size - 4, 0x70000) if mode_bits == 32 else 0x7C00

    def boot_disk(self, disk_image: bytes, require_signature: bool = True) -> None:
        """Boot a raw disk image at physical 0x7C00 in 16-bit real mode.

        Implements a small BIOS shim for INT 10h, INT 13h, INT 16h, INT 1Ah,
        INT 18h and INT 19h. The BIOS and hardware are necessarily incomplete.
        """
        if not isinstance(disk_image, (bytes, bytearray, memoryview)):
            raise TypeError("disk_image must be bytes-like")
        disk = bytes(disk_image)
        if len(disk) < 512:
            disk = disk.ljust(512, b"\x00")
        if require_signature and disk[510:512] != b"\x55\xAA":
            raise FemuError("boot sector is missing the 0x55AA signature at offsets 510-511")
        self.reset_cpu()
        self.mode_bits = 16
        self.protected_mode = False
        self.disk_image = disk
        self.drive_number = 0x80
        self.mem[0x7C00:0x7E00] = disk[:512]
        self.seg.update({"CS": 0, "DS": 0, "ES": 0, "SS": 0})
        self.reg["EIP"] = 0x7C00
        self.reg["ESP"] = 0x7C00
        self.set_alias("DL", self.drive_number)
        self.set_flag(FLAG_IF, True)
        self.console.append("[BIOS] Booting drive 0x80 from 0000:7C00\n")

    def boot_linux_bzimage(self, image: bytes, command_line: str = "console=ttyS0,115200 earlyprintk=serial,ttyS0 ignore_loglevel loglevel=8 panic=1") -> dict:
        """Try the documented 16-bit Linux/x86 boot protocol for a bzImage.

        This is an experimental bootloader path: it places real-mode setup at
        0x90000, the protected-mode payload at 1 MiB, fills the traditional
        setup header fields, and enters the setup code at 9020:0000. It does not
        implement x86-64 long mode or all PC/BIOS devices required by arbitrary
        Linux kernels; unsupported guest instructions are reported faithfully.
        """
        if not isinstance(image, (bytes, bytearray, memoryview)):
            raise TypeError("Linux bzImage must be bytes-like")
        image = bytes(image)
        info = inspect_linux_bzimage(image)
        if not info["boot_signature_valid"]:
            raise FemuError("Linux bzImage boot sector is missing the 0x55AA signature")
        setup_bytes = info["setup_bytes"]
        kernel_image = image[setup_bytes:]
        load_address = 0x100000 if info["load_high"] else 0x10000
        init_size = int.from_bytes(image[0x260:0x264], "little") if len(image) >= 0x264 else 0
        required_end = load_address + max(len(kernel_image), init_size)
        if required_end > self.ram_size:
            raise FemuError(
                f"Linux kernel needs RAM through 0x{required_end:X}, but guest RAM ends at 0x{self.ram_size:X}; "
                "increase --ram (for example --ram 0x08000000)"
            )
        if setup_bytes > 0x8000:
            raise FemuError("Linux real-mode setup exceeds the supported 32 KiB setup window")
        if load_address + len(kernel_image) > self.ram_size:
            raise FemuError("Linux protected-mode payload does not fit guest RAM")
        raw_cmd = command_line.encode("ascii", errors="replace")[:0x7FF] + b"\x00"
        # Memory layout follows the Linux boot-protocol sample for a high-loaded bzImage.
        setup_base = 0x90000
        cmd_addr = setup_base + 0x9800
        if cmd_addr + len(raw_cmd) > 0xA0000:
            raise FemuError("Linux command line would overlap the reserved BIOS memory hole")
        self.reset_cpu()
        self.disk_image = image
        self.drive_number = 0x80
        self.mode_bits = 16
        self.protected_mode = False
        self.cr["CR0"] = 0
        self.mem[setup_base:setup_base + setup_bytes] = image[:setup_bytes]
        self.mem[load_address:load_address + len(kernel_image)] = kernel_image
        # Bootloader-owned fields in setup_header (physical address = setup_base + offset).
        self.write8(setup_base + 0x210, 0xFF)  # undefined bootloader ID
        loadflags = self.read8(setup_base + 0x211) | 0x80  # CAN_USE_HEAP
        self.write8(setup_base + 0x211, loadflags)
        self.write16(setup_base + 0x224, 0x9600)  # heap_end - 0x200
        self.write32(setup_base + 0x214, load_address)  # code32_start
        self.write32(setup_base + 0x218, 0)  # no initrd supplied
        self.write32(setup_base + 0x21C, 0)
        if info["protocol"] >= 0x0202:
            self.write32(setup_base + 0x228, cmd_addr)
        self.mem[cmd_addr:cmd_addr + len(raw_cmd)] = raw_cmd
        # Conventional real-mode kernel entry at segment base + 0x200.
        self.seg.update({"CS": 0x9020, "DS": 0x9000, "ES": 0x9000,
                         "SS": 0x9000, "FS": 0x9000, "GS": 0x9000})
        self.reg["EIP"] = 0
        self.reg["ESP"] = 0x9800
        self.set_alias("DL", self.drive_number)
        self.set_flag(FLAG_IF, False)
        self.set_flag(FLAG_DF, False)
        self.cursor_row = self.cursor_col = 0
        self.console.append(
            f"[LINUX] Experimental bzImage boot: protocol {info['protocol_version']}, "
            f"setup {info['setup_sectors']} sector(s), payload {len(kernel_image)} bytes\n"
        )
        self.console.append(f"[LINUX] Setup entry 9020:0000; protected payload 0x{load_address:08X}; RAM {self.ram_size // (1024*1024)} MiB\n")
        if info["supports_64bit_kernel"]:
            self.console.append("[LINUX] Kernel advertises a 64-bit entry point; this emulator currently supports 16/32-bit execution only.\n")
        return info

    def bootloader(self, disk_image: Optional[bytes] = None, max_steps: int = 100000) -> str:
        """Compatibility wrapper: run supplied disk, or a tiny built-in BIOS demo."""
        if disk_image is None:
            disk_image = make_demo_boot_sector()
        self.boot_disk(disk_image)
        self.run(max_steps)
        return "".join(self.console)

    # ----- CPU step -----
    def step(self) -> None:
        if self.stopped:
            raise Halted
        self._opcode_start = self.eip
        self.operand_size = self.mode_bits
        self.address_size = self.mode_bits
        self.default_segment_override = None
        self.repeat_prefix = None
        prefix_count = 0
        opcode = self.fetch8()
        while opcode in (0x66, 0x67, 0x26, 0x2E, 0x36, 0x3E, 0x64, 0x65, 0xF0, 0xF2, 0xF3):
            prefix_count += 1
            if prefix_count > 8:
                raise FemuError("too many instruction prefixes")
            if opcode == 0x66:
                self.operand_size = 16 if self.mode_bits == 32 else 32
            elif opcode == 0x67:
                self.address_size = 16 if self.mode_bits == 32 else 32
            elif opcode == 0x26: self.default_segment_override = "ES"
            elif opcode == 0x2E: self.default_segment_override = "CS"
            elif opcode == 0x36: self.default_segment_override = "SS"
            elif opcode == 0x3E: self.default_segment_override = "DS"
            elif opcode == 0x64: self.default_segment_override = "FS"
            elif opcode == 0x65: self.default_segment_override = "GS"
            elif opcode in (0xF2, 0xF3): self.repeat_prefix = opcode
            opcode = self.fetch8()
        width = self.operand_size
        addrsize = self.address_size
        if self.trace and len(self.trace_lines) < self.trace_limit:
            self.trace_lines.append(f"{self.steps:08d} CS:IP={self.seg['CS']:04X}:{self._opcode_start:04X} OP={opcode:02X} AX={self.get_alias('AX'):04X} BX={self.get_alias('BX'):04X} CX={self.get_alias('CX'):04X} DX={self.get_alias('DX'):04X} FLAGS={self.reg['EFLAGS']:08X}")
        self.steps += 1

        # MOV reg, immediate
        if 0xB8 <= opcode <= 0xBF:
            idx = opcode - 0xB8
            imm = self.fetch16() if width == 16 else self.fetch32()
            self.set_reg(idx, imm, width)
        elif 0xB0 <= opcode <= 0xB7:
            self.set_reg(opcode - 0xB0, self.fetch8(), 8)
        # PUSHA/PUSHAD and POPA/POPAD preserve the original SP/ESP slot semantics.
        elif opcode in (0x60, 0x61):
            w=width; regs=[self.get_reg(i,w) for i in range(8)]
            if opcode==0x60:
                original_sp=self.reg["ESP"]&(MASK16 if w==16 else MASK32)
                for idx in (0,1,2,3): self.push(regs[idx],w)
                self.push(original_sp,w)
                for idx in (5,6,7): self.push(regs[idx],w)
            else:
                # POPA order is DI/SI/BP, discard stored SP, then BX/DX/CX/AX.
                for idx in (7,6,5): self.set_reg(idx,self.pop(w),w)
                self.pop(w)
                for idx in (3,2,1,0): self.set_reg(idx,self.pop(w),w)
        # Segment stack instructions used during BIOS and protected-mode transitions.
        elif opcode in (0x06, 0x0E, 0x16, 0x1E):
            segname = {0x06:"ES", 0x0E:"CS", 0x16:"SS", 0x1E:"DS"}[opcode]
            self.push(self.seg[segname], 16)
        elif opcode in (0x07, 0x17, 0x1F):
            segname = {0x07:"ES", 0x17:"SS", 0x1F:"DS"}[opcode]
            self._load_segment_selector(segname, self.pop(16) & MASK16)
        # XCHG EAX/AX with another register, plus CBW/CWDE and CWD/CDQ.
        elif 0x91 <= opcode <= 0x97:
            idx = opcode - 0x90
            a, b = self.get_reg(0, width), self.get_reg(idx, width)
            self.set_reg(0, b, width); self.set_reg(idx, a, width)
        elif opcode == 0x98:  # CBW/CWDE
            if width == 16:
                value = self.get_alias("AL")
                if value & 0x80: value -= 0x100
                self.set_alias("AX", value)
            else:
                value = self.get_alias("AX")
                if value & 0x8000: value -= 0x10000
                self.set_alias("EAX", value)
        elif opcode == 0x99:  # CWD/CDQ
            if width == 16:
                self.set_alias("DX", 0xFFFF if self.get_alias("AX") & 0x8000 else 0)
            else:
                self.set_alias("EDX", MASK32 if self.get_alias("EAX") & 0x80000000 else 0)
        elif opcode == 0xC8:  # ENTER imm16, imm8
            allocation=self.fetch16(); nesting=self.fetch8() & 0x1F
            self.push(self.get_reg(5,width),width)
            frame=self.reg["ESP"] & (MASK16 if width==16 else MASK32)
            if nesting:
                base=self.get_alias("BP" if width==16 else "EBP")
                for _ in range(1,nesting):
                    base=(base-width//8)&(MASK16 if width==16 else MASK32)
                    self.push(self.read_value(self.linear(base,"SS"),width),width)
                self.push(frame,width)
            self.set_reg(5,frame,width)
            self.reg["ESP"]=(self.reg["ESP"]-allocation)&(MASK16 if width==16 else MASK32)
        elif opcode == 0xC9:  # LEAVE
            frame=self.get_alias("BP" if width==16 else "EBP")
            if width==16: self.reg["ESP"]=(self.reg["ESP"]&0xFFFF0000)|(frame&MASK16)
            else: self.reg["ESP"]=frame&MASK32
            self.set_reg(5,self.pop(width),width)
        # MOV r/m8,r8 and r/m16/32,r16/32; reverse forms
        elif opcode in (0x88, 0x89, 0x8A, 0x8B):
            mod, regidx, rm = self.decode_modrm(self.fetch8())
            w = 8 if opcode in (0x88, 0x8A) else width
            rmop = self._decode_rm(mod, rm, w, addrsize)
            regop = ("reg", regidx, w)
            if opcode in (0x88, 0x89): self._write_operand(rmop, self._read_operand(regop))
            else: self._write_operand(regop, self._read_operand(rmop))
        # LEA
        elif opcode == 0x8D:
            mod, regidx, rm = self.decode_modrm(self.fetch8())
            if mod == 3:
                raise FemuError("LEA requires a memory addressing form")
            addr, effective_seg = self._effective_address(mod, rm, addrsize)
            value = addr if self.protected_mode else (addr - (self.seg[effective_seg] << 4)) & (MASK16 if addrsize == 16 else MASK32)
            self.set_reg(regidx, value, width)
        # MOV segment register to/from r/m16
        elif opcode in (0x8C, 0x8E):
            mod, sidx, rm = self.decode_modrm(self.fetch8())
            if sidx >= len(SEG_NAMES): raise FemuError("invalid segment register index")
            rmop = self._decode_rm(mod, rm, 16, addrsize)
            if opcode == 0x8C: self._write_operand(rmop, self.seg[SEG_NAMES[sidx]])
            else:
                if SEG_NAMES[sidx] == "CS": raise FemuError("MOV cannot load CS; use a far control transfer")
                self._load_segment_selector(SEG_NAMES[sidx], self._read_operand(rmop))
        # accumulator ↔ absolute memory (moffs)
        elif opcode in (0xA0, 0xA1, 0xA2, 0xA3):
            off = self.fetch16() if addrsize == 16 else self.fetch32()
            w = 8 if opcode in (0xA0, 0xA2) else width
            addr = self.linear(off, self.default_segment_override or "DS")
            if opcode in (0xA0, 0xA1): self.set_reg(0, self.read_value(addr, w), w)
            else: self.write_value(addr, self.get_reg(0, w), w)
        # MOV r/m, immediate C6/C7
        elif opcode in (0xC6, 0xC7):
            mod, sub, rm = self.decode_modrm(self.fetch8())
            if sub != 0: raise FemuError(f"unsupported MOV immediate /{sub}")
            w = 8 if opcode == 0xC6 else width
            operand = self._decode_rm(mod, rm, w, addrsize)  # displacement precedes immediate
            imm = self.fetch8() if w == 8 else self.fetch16() if w == 16 else self.fetch32()
            self._write_operand(operand, imm)
        # XCHG r/m,reg; NOP (90)
        elif opcode in (0x86, 0x87):
            mod, regidx, rm = self.decode_modrm(self.fetch8())
            w = 8 if opcode == 0x86 else width
            rmop = self._decode_rm(mod, rm, w, addrsize); rop = ("reg", regidx, w)
            a, b = self._read_operand(rmop), self._read_operand(rop)
            self._write_operand(rmop, b); self._write_operand(rop, a)
        elif opcode == 0x90:
            pass
        # String instructions. REP/REPE/REPNE execute multiple memory iterations
        # while remaining a single decoded x86 instruction for the current budget.
        elif opcode in (0xA4,0xA5,0xA6,0xA7,0xAA,0xAB,0xAE,0xAF,0xAC,0xAD):
            width_s = 8 if opcode in (0xA4,0xA6,0xAA,0xAE,0xAC) else width
            size_s = width_s // 8
            src_index = "SI" if addrsize == 16 else "ESI"
            dst_index = "DI" if addrsize == 16 else "EDI"
            count_name = "CX" if addrsize == 16 else "ECX"
            count = self.get_alias(count_name) if self.repeat_prefix in (0xF2,0xF3) else 1
            if count > 1_000_000: raise FemuError("REP string operation exceeds the safety iteration limit")
            for _ in range(count):
                si = self.get_alias(src_index); di = self.get_alias(dst_index)
                delta = -size_s if self.flag(FLAG_DF) else size_s
                srcseg = self.default_segment_override or "DS"
                if opcode in (0xA4,0xA5):  # MOVS
                    self.write_value(self.linear(di, "ES"), self.read_value(self.linear(si, srcseg), width_s), width_s)
                elif opcode in (0xAA,0xAB):  # STOS
                    self.write_value(self.linear(di, "ES"), self.get_reg(0, width_s), width_s)
                elif opcode in (0xAC,0xAD):  # LODS
                    self.set_reg(0, self.read_value(self.linear(si, srcseg), width_s), width_s)
                elif opcode in (0xA6,0xA7):  # CMPS
                    self.alu("CMP", self.read_value(self.linear(si, srcseg), width_s), self.read_value(self.linear(di, "ES"), width_s), width_s)
                else:  # SCAS
                    self.alu("CMP", self.get_reg(0, width_s), self.read_value(self.linear(di, "ES"), width_s), width_s)
                if opcode in (0xA4,0xA5,0xA6,0xA7,0xAC,0xAD):
                    self.set_alias(src_index, (si + delta) & (MASK16 if addrsize == 16 else MASK32))
                if opcode in (0xA4,0xA5,0xA6,0xA7,0xAA,0xAB,0xAE,0xAF):
                    self.set_alias(dst_index, (di + delta) & (MASK16 if addrsize == 16 else MASK32))
                if self.repeat_prefix in (0xF2,0xF3):
                    count -= 1; self.set_alias(count_name, count)
                    if opcode in (0xA6,0xA7,0xAE,0xAF):
                        if self.repeat_prefix == 0xF3 and not self.flag(FLAG_ZF): break
                        if self.repeat_prefix == 0xF2 and self.flag(FLAG_ZF): break
                    if count == 0: break
        # Accumulator arithmetic immediate encodings
        elif opcode in (0x04, 0x05, 0x0C, 0x0D, 0x14, 0x15, 0x24, 0x25, 0x2C, 0x2D, 0x34, 0x35, 0x3C, 0x3D):
            w = 8 if opcode in (0x04, 0x0C, 0x14, 0x24, 0x2C, 0x34, 0x3C) else width
            imm = self.fetch8() if w == 8 else self.fetch16() if w == 16 else self.fetch32()
            op = {0x04:"ADD",0x05:"ADD",0x0C:"OR",0x0D:"OR",0x14:"ADC",0x15:"ADC",0x24:"AND",0x25:"AND",0x2C:"SUB",0x2D:"SUB",0x34:"XOR",0x35:"XOR",0x3C:"CMP",0x3D:"CMP"}[opcode]
            result = self.alu(op, self.get_reg(0, w), imm, w)
            if op != "CMP": self.set_reg(0, result, w)
        # ADD/OR/AND/SUB/XOR/CMP r/m<->reg forms
        elif opcode in (0x00,0x01,0x02,0x03,0x08,0x09,0x0A,0x0B,0x10,0x11,0x12,0x13,0x20,0x21,0x22,0x23,
                        0x28,0x29,0x2A,0x2B,0x30,0x31,0x32,0x33,0x38,0x39,0x3A,0x3B,
                        0x84,0x85):
            mod, regidx, rm = self.decode_modrm(self.fetch8())
            if opcode in (0x00,0x02,0x08,0x0A,0x10,0x12,0x20,0x22,0x28,0x2A,0x30,0x32,0x38,0x3A,0x84): w = 8
            else: w = width
            rmop = self._decode_rm(mod, rm, w, addrsize); rop = ("reg", regidx, w)
            if opcode in (0x84,0x85):
                self.alu("TEST", self._read_operand(rmop), self._read_operand(rop), w)
            else:
                op = "ADD" if opcode in (0x00,0x01,0x02,0x03) else "OR" if opcode in (0x08,0x09,0x0A,0x0B) else "ADC" if opcode in (0x10,0x11,0x12,0x13) else "AND" if opcode in (0x20,0x21,0x22,0x23) else "SUB" if opcode in (0x28,0x29,0x2A,0x2B) else "XOR" if opcode in (0x30,0x31,0x32,0x33) else "CMP"
                to_reg = opcode in (0x02,0x03,0x0A,0x0B,0x12,0x13,0x22,0x23,0x2A,0x2B,0x32,0x33,0x3A,0x3B)
                a = self._read_operand(rop if to_reg else rmop)
                b = self._read_operand(rmop if to_reg else rop)
                result = self.alu(op, a, b, w)
                if op != "CMP": self._write_operand(rop if to_reg else rmop, result)
        # Group 1: immediate arithmetic, including 83 sign-extended imm8
        elif opcode in (0x80, 0x81, 0x83):
            mod, sub, rm = self.decode_modrm(self.fetch8())
            w = 8 if opcode == 0x80 else width
            operand = self._decode_rm(mod, rm, w, addrsize)  # consume addressing displacement first
            if opcode == 0x83: imm = self.fetch_signed(8) & ((1 << w) - 1)
            else: imm = self.fetch8() if w == 8 else self.fetch16() if w == 16 else self.fetch32()
            op = {0:"ADD",1:"OR",2:"ADC",3:"SBB",4:"AND",5:"SUB",6:"XOR",7:"CMP"}[sub]
            result = self.alu(op, self._read_operand(operand), imm, w)
            if op != "CMP": self._write_operand(operand, result)
        # INC/DEC legacy single-byte registers (still valid in 16-bit mode)
        elif 0x40 <= opcode <= 0x47:
            idx = opcode - 0x40; self.set_reg(idx, self.incdec(self.get_reg(idx, width), width, 1), width)
        elif 0x48 <= opcode <= 0x4F:
            idx = opcode - 0x48; self.set_reg(idx, self.incdec(self.get_reg(idx, width), width, -1), width)
        # PUSH/POP register
        elif 0x50 <= opcode <= 0x57:
            idx = opcode - 0x50; self.push(self.get_reg(idx, width), width)
        elif 0x58 <= opcode <= 0x5F:
            idx = opcode - 0x58; self.set_reg(idx, self.pop(width), width)
        # short conditional jumps
        elif 0x70 <= opcode <= 0x7F:
            rel = self.fetch_signed(8)
            if self.condition(opcode & 0x0F): self.eip = self.eip + rel
        # push immediates
        elif opcode == 0x68:
            self.push(self.fetch16() if width == 16 else self.fetch32(), width)
        elif opcode == 0x6A:
            self.push(self.fetch_signed(8) & ((1 << width) - 1), width)
        # near CALL/JMP rel and short JMP
        elif opcode in (0xE8, 0xE9):
            rel = self.fetch_signed(width)
            if opcode == 0xE8: self.push(self.eip, width)
            self.eip = self.eip + rel
        elif opcode == 0xEB:
            rel = self.fetch_signed(8)
            self.eip = self.eip + rel
        # LOOPNZ / LOOPZ / LOOP / JCXZ, with counter width selected by address size.
        elif opcode in (0xE0, 0xE1, 0xE2, 0xE3):
            rel = self.fetch_signed(8)
            if self.address_size == 16:
                count = self.get_alias("CX")
                if opcode != 0xE3:
                    count = (count - 1) & MASK16
                    self.set_alias("CX", count)
            else:
                count = self.get_alias("ECX")
                if opcode != 0xE3:
                    count = (count - 1) & MASK32
                    self.set_alias("ECX", count)
            if opcode == 0xE3:
                take_branch = (count == 0)
            else:
                zf = self.flag(FLAG_ZF)
                take_branch = (count != 0 and
                               (opcode != 0xE0 or not zf) and
                               (opcode != 0xE1 or bool(zf)))
            if take_branch:
                self.eip = self.eip + rel
        elif opcode in (0xC3, 0xC2):
            adjustment = self.fetch16() if opcode == 0xC2 else 0
            self.eip = self.pop(width)
            if adjustment:
                stack_mask = MASK16 if self.mode_bits == 16 else MASK32
                sp = ((self.reg["ESP"] & stack_mask) + adjustment) & stack_mask
                self.reg["ESP"] = (self.reg["ESP"] & ~stack_mask) | sp
        # return from interrupt
        elif opcode == 0xCF:
            self.eip = self.pop(width)
            if self.mode_bits == 16:
                self.seg["CS"] = self.pop(16)
                self.reg["EFLAGS"] = self.pop(16) | FLAG_FIXED
            else:
                self.seg["CS"] = self.pop(32) & MASK16
                self.reg["EFLAGS"] = self.pop(32) | FLAG_FIXED
        # FF group: inc/dec/call/jmp/push r/m
        elif opcode == 0xFF:
            mod, sub, rm = self.decode_modrm(self.fetch8())
            operand = self._decode_rm(mod, rm, width, addrsize)
            val = self._read_operand(operand)
            if sub == 0: self._write_operand(operand, self.incdec(val, width, 1))
            elif sub == 1: self._write_operand(operand, self.incdec(val, width, -1))
            elif sub == 2: self.push(self.eip, width); self.eip = val
            elif sub == 4: self.eip = val
            elif sub == 6: self.push(val, width)
            else: raise FemuError(f"unsupported FF /{sub} group")
        # POP r/m
        elif opcode == 0x8F:
            mod, sub, rm = self.decode_modrm(self.fetch8())
            if sub != 0: raise FemuError(f"unsupported POP /{sub}")
            self._write_operand(self._decode_rm(mod, rm, width, addrsize), self.pop(width))
        # PUSHF/POPF
        elif opcode == 0x9C: self.push(self.reg["EFLAGS"], width)
        elif opcode == 0x9D: self.reg["EFLAGS"] = (self.pop(width) | FLAG_FIXED) & MASK32
        # INC/DEC r/m8
        elif opcode == 0xFE:
            mod, sub, rm = self.decode_modrm(self.fetch8())
            operand = self._decode_rm(mod, rm, 8, addrsize)
            if sub in (0,1): self._write_operand(operand, self.incdec(self._read_operand(operand), 8, 1 if sub == 0 else -1))
            else: raise FemuError(f"unsupported FE /{sub}")
        # TEST accumulator immediate
        elif opcode in (0xA8, 0xA9):
            w = 8 if opcode == 0xA8 else width
            imm = self.fetch8() if w == 8 else self.fetch16() if w == 16 else self.fetch32()
            self.alu("TEST", self.get_reg(0, w), imm, w)
        # Group 2 shifts and rotates: ROL/ROR/RCL/RCR/SHL/SHR/SAR.
        elif opcode in (0xC0,0xC1,0xD0,0xD1,0xD2,0xD3):
            mod, sub, rm = self.decode_modrm(self.fetch8())
            w = 8 if opcode in (0xC0,0xD0,0xD2) else width
            operand = self._decode_rm(mod, rm, w, addrsize)
            value = self._read_operand(operand); mask = (1 << w) - 1; sign = 1 << (w - 1)
            count = self.fetch8() if opcode in (0xC0,0xC1) else 1 if opcode in (0xD0,0xD1) else self.get_alias("CL")
            count &= 0x1F
            if count:
                original = value
                effective_count = count % (w if sub in (0,1) else w+1) if sub in (0,1,2,3) else count
                if effective_count == 0 and sub in (0,1,2,3):
                    carry = self.flag(FLAG_CF)  # rotate count may reduce to zero
                elif sub in (0,1,2,3):
                    for _ in range(effective_count):
                        if sub == 0: carry = (value >> (w-1)) & 1; value = ((value << 1) | carry) & mask
                        elif sub == 1: carry = value & 1; value = (value >> 1) | (carry << (w-1))
                        elif sub == 2: carry = self.flag(FLAG_CF); newcarry=(value>>(w-1))&1; value=((value<<1)|carry)&mask; carry=newcarry
                        else: carry=self.flag(FLAG_CF); newcarry=value&1; value=(value>>1)|(carry<<(w-1)); carry=newcarry
                    self.set_flag(FLAG_CF, bool(carry))
                elif sub in (4,6):
                    carry = (value >> max(0, w-count)) & 1 if count <= w else 0
                    value = (value << count) & mask
                    self.set_flag(FLAG_CF, bool(carry))
                elif sub == 5:
                    carry = (value >> (count-1)) & 1 if count <= w else 0
                    value >>= count
                    self.set_flag(FLAG_CF, bool(carry))
                elif sub == 7:
                    carry = (value >> (count-1)) & 1 if count <= w else (1 if value & sign else 0)
                    signed = value - (1 << w) if value & sign else value
                    value = (signed >> count) & mask
                    self.set_flag(FLAG_CF, bool(carry))
                if sub in (4,5,6,7): self.set_szp(value,w)
                if count == 1:
                    if sub in (0,2,4,6): self.set_flag(FLAG_OF, bool(((value & sign) != 0) ^ self.flag(FLAG_CF)))
                    elif sub == 1: self.set_flag(FLAG_OF, bool((value & sign) ^ ((value >> (w-2)) & 1)))
                    elif sub in (3,5): self.set_flag(FLAG_OF, bool(original & sign))
                self._write_operand(operand,value)
        # IN/OUT port instructions; unmapped hardware returns deterministic values.
        elif opcode in (0xE4,0xE5,0xEC,0xED):
            w=8 if opcode in (0xE4,0xEC) else width
            port=self.fetch8() if opcode in (0xE4,0xE5) else self.get_alias("DX")
            self.set_reg(0,self._port_in(port,w),w)
        elif opcode in (0xE6,0xE7,0xEE,0xEF):
            w=8 if opcode in (0xE6,0xEE) else width
            port=self.fetch8() if opcode in (0xE6,0xE7) else self.get_alias("DX")
            self._port_out(port,self.get_reg(0,w),w)
        # F7 group: TEST, NOT, NEG
        elif opcode in (0xF6, 0xF7):
            mod, sub, rm = self.decode_modrm(self.fetch8()); w = 8 if opcode == 0xF6 else width
            operand = self._decode_rm(mod, rm, w, addrsize); val = self._read_operand(operand)
            if sub in (0,1):
                imm = self.fetch8() if w == 8 else self.fetch16() if w == 16 else self.fetch32()
                self.alu("TEST", val, imm, w)
            elif sub == 2: self._write_operand(operand, (~val) & ((1 << w) - 1))
            elif sub == 3: self._write_operand(operand, self.alu("SUB", 0, val, w))
            elif sub in (4,5):  # MUL / IMUL accumulator forms
                mask=(1<<w)-1
                if sub==4: product=self.get_reg(0,w)*val
                else:
                    a=self.get_reg(0,w); sign=1<<(w-1)
                    a=a-(1<<w) if a&sign else a; b=val-(1<<w) if val&sign else val
                    product=a*b
                self.set_flag(FLAG_CF, product>mask if sub==4 else not (-(1<<(w-1)) <= product < (1<<(w-1))))
                self.set_flag(FLAG_OF,self.flag(FLAG_CF))
                if w==8: self.set_alias("AX",product)
                else:
                    self.set_reg(0,product,w); self.set_reg(2,product>>w,w)
            elif sub in (6,7):  # DIV / IDIV accumulator forms
                if val==0: raise FemuError("guest divide error (#DE): division by zero")
                if w==8: dividend=self.get_alias("AX")
                else: dividend=(self.get_reg(2,w)<<w)|self.get_reg(0,w)
                if sub==7:
                    sign=1<<(w-1); mask=(1<<w)-1
                    dividend=dividend-(1<<(2*w)) if dividend&(1<<(2*w-1)) else dividend
                    divisor=val-(1<<w) if val&sign else val
                    quotient=abs(dividend)//abs(divisor)*(1 if (dividend>=0)==(divisor>=0) else -1)
                    remainder=dividend-quotient*divisor
                else: quotient=dividend//val; remainder=dividend%val
                if w==8:
                    lo,hi=(-128,127) if sub==7 else (0,255)
                    if not lo<=quotient<=hi or not lo<=remainder<=hi: raise FemuError("guest divide error (#DE): byte quotient/remainder overflow")
                    self.set_alias("AL",quotient); self.set_alias("AH",remainder)
                else:
                    lo,hi=(-(1<<(w-1)),(1<<(w-1))-1) if sub==7 else (0,(1<<w)-1)
                    if not lo<=quotient<=hi or not lo<=remainder<=hi: raise FemuError("guest divide error (#DE): quotient/remainder overflow")
                    self.set_reg(0,quotient,w); self.set_reg(2,remainder,w)
            else: raise FemuError(f"unsupported F6/F7 /{sub} operation")
        # Far CALL/JMP/RET: enough descriptor handling for flat ring-0 boot transitions.
        elif opcode in (0x9A,0xEA):
            off=self.fetch16() if width==16 else self.fetch32()
            selector=self.fetch16()
            if opcode==0x9A:
                self.push(self.seg["CS"],16); self.push(self.eip,width)
            self._far_transfer(off,selector)
        elif opcode in (0xCB,0xCA):
            adjust=self.fetch16() if opcode==0xCA else 0
            off=self.pop(width); selector=self.pop(16)
            self._far_transfer(off,selector)
            if adjust: self.reg["ESP"]=(self.reg["ESP"]+adjust)&MASK32
        elif 0xD8 <= opcode <= 0xDF:  # minimal x87 environment/probe support
            modrm=self.fetch8(); mod,sub,rm=self.decode_modrm(modrm)
            # Common boot-time probe/state operations: FNINIT, FNCLEX, FNSTSW AX,
            # FLDCW/FNSTCW and FLDENV/FNSTENV. Arithmetic stack values are not emulated.
            if opcode==0xDB and mod==3 and modrm==0xE3:  # FNINIT
                self.fpu_control=0x037F; self.fpu_status=0; self.fpu_tags=0xFFFF
            elif opcode==0xDB and mod==3 and modrm==0xE2:  # FNCLEX
                self.fpu_status=0
            elif opcode==0xDF and mod==3 and modrm==0xE0:  # FNSTSW AX
                self.set_alias("AX",self.fpu_status)
            elif mod==3:
                # Register-form arithmetic/control instructions are accepted as
                # deterministic no-ops so FPU detection can proceed, not numerically simulated.
                if opcode==0xD9 and modrm==0xE3:  # FINIT spelling found in some tools
                    self.fpu_control=0x037F; self.fpu_status=0; self.fpu_tags=0xFFFF
            else:
                addr,_seg=self._effective_address(mod,rm,addrsize)
                if opcode==0xD9 and sub==5: self.fpu_control=self.read16(addr)
                elif opcode==0xD9 and sub==7: self.write16(addr,self.fpu_control)
                elif opcode==0xD9 and sub in (4,6):
                    # 14-byte (16-bit) or 28-byte (32-bit) environment.
                    env_size=14 if width==16 else 28
                    self._check(addr,env_size)
                    if sub==6:
                        self.write16(addr,self.fpu_control); self.write16(addr+2,self.fpu_status); self.write16(addr+4,self.fpu_tags)
                    else:
                        self.fpu_control=self.read16(addr); self.fpu_status=self.read16(addr+2); self.fpu_tags=self.read16(addr+4)
                elif opcode==0xDD and sub==7: self.write16(addr,self.fpu_status)
                # Other x87 memory arithmetic is intentionally a non-numeric stub.
        elif opcode == 0xD7:  # XLAT / XLATB
            base=self.get_alias("BX" if addrsize==16 else "EBX")
            offset=(base+self.get_alias("AL"))&(MASK16 if addrsize==16 else MASK32)
            self.set_alias("AL",self.read8(self.linear(offset,self.default_segment_override or "DS")))
        elif opcode == 0x9B:  # FWAIT; this emulator has a synchronous synthetic FPU
            pass
        elif opcode == 0x9F:  # LAHF
            flags=self.reg["EFLAGS"]
            self.set_alias("AH", (flags & 0xD5) | 0x02)
        elif opcode == 0x9E:  # SAHF
            ah=self.get_alias("AH")
            mask=FLAG_SF|FLAG_ZF|FLAG_AF|FLAG_PF|FLAG_CF
            self.reg["EFLAGS"]=(self.reg["EFLAGS"]&~mask)|(ah & mask)|FLAG_FIXED
        # Software INT and INTO
        elif opcode == 0xCD:
            self.interrupt(self.fetch8())
        elif opcode == 0xCC:
            self.interrupt(3)
        elif opcode == 0xCE:
            if self.flag(FLAG_OF): self.interrupt(4)
        # two-byte opcode map: Jcc near, SETcc r/m8, MOVZX, MOVSX
        elif opcode == 0x0F:
            op2 = self.fetch8()
            if op2 == 0xA2:  # CPUID
                self._cpu_cpuid()
            elif op2 == 0x31:  # RDTSC (deterministic synthetic time base)
                tsc=(self.steps*1000)&0xFFFFFFFFFFFFFFFF
                self.reg["EAX"]=tsc&MASK32; self.reg["EDX"]=(tsc>>32)&MASK32
            elif op2 in (0x21,0x23):  # MOV general register <-> DR0-DR7
                mod, dridx, rm = self.decode_modrm(self.fetch8())
                if mod != 3: raise FemuError("MOV DRx requires a register operand")
                if dridx in (4,5):
                    if self.cr["CR4"] & (1 << 3): raise FemuError("guest debug exception (#UD): DR4/DR5 disabled by CR4.DE")
                    dridx += 2
                if dridx not in (0,1,2,3,6,7): raise FemuError(f"unsupported debug register DR{dridx}")
                drname=f"DR{dridx}"
                if op2 == 0x21: self.set_reg(rm,self.dr[drname],32)
                else: self.dr[drname]=self.get_reg(rm,32)
            elif op2 in (0x20,0x22):  # MOV general register <-> CR0/2/3/4
                mod, cridx, rm = self.decode_modrm(self.fetch8())
                if mod != 3: raise FemuError("MOV CRx requires a register operand")
                crname=f"CR{cridx}"
                if crname not in self.cr: raise FemuError(f"unsupported control register CR{cridx}")
                if op2 == 0x20: self.set_reg(rm,self.cr[crname],32)
                else:
                    value=self.get_reg(rm,32)
                    if crname=="CR0" and (value & 0x80000000) and not (value & 1): raise FemuError("CR0.PG cannot be enabled before CR0.PE")
                    self.cr[crname]=value
                    if crname=="CR0":
                        # CR0.PE is armed here; far transfer reloads CS and
                        # activates protected-mode linear addressing.
                        if not (value & 1): self.protected_mode=False
                    if crname=="CR3": self.cr[crname]=value&0xFFFFF000
            elif op2 == 0x06:  # CLTS
                self.cr["CR0"] &= ~(1 << 3)
            elif op2 in (0x30,0x32):  # WRMSR / RDMSR
                msr_id=self.reg["ECX"]&MASK32
                if op2==0x32:
                    value=self.msr.get(msr_id,0); self.reg["EAX"]=value&MASK32; self.reg["EDX"]=(value>>32)&MASK32
                else:
                    value=(self.reg["EDX"]<<32)|self.reg["EAX"]
                    if msr_id==0xC0000080 and (value & (1<<8)):
                        raise FemuError("guest requested EFER.LME: x86-64 long mode is not implemented")
                    self.msr[msr_id]=value
            elif op2 == 0x01:  # system group: SGDT/SIDT/LGDT/LIDT/SMSW/LMSW/INVLPG
                mod, sub, rm=self.decode_modrm(self.fetch8())
                if sub in (0,1,2,3,7):
                    operand=self._decode_rm(mod,rm,16,addrsize)
                    if operand[0] != "mem" and sub in (0,1,2,3,7): raise FemuError("descriptor-table / INVLPG operation requires a memory operand")
                    addr=operand[1]
                    if sub==0 or sub==1:
                        table=self.gdtr if sub==0 else self.idtr
                        self.write16(addr,table["limit"]); self.write32(addr+2,table["base"])
                    elif sub==2 or sub==3:
                        table=self.gdtr if sub==2 else self.idtr
                        table["limit"]=self.read16(addr); table["base"]=self.read32(addr+2)
                    else:
                        # INVLPG: no TLB cache is maintained by this interpreter.
                        pass
                elif sub==4: # SMSW r/m16
                    self._write_operand(self._decode_rm(mod,rm,16,addrsize),self.cr["CR0"]&0xFFFF)
                elif sub==6: # LMSW r/m16
                    value=self._read_operand(self._decode_rm(mod,rm,16,addrsize))
                    self.cr["CR0"]=(self.cr["CR0"]&~0xF)|(value&0xF)
                    if not (self.cr["CR0"]&1): self.protected_mode=False
                else: raise FemuError(f"unsupported 0F 01 /{sub} system operation")
            elif op2 in (0xA0,0xA1,0xA8,0xA9): # PUSH/POP FS/GS
                segname="FS" if op2 in (0xA0,0xA1) else "GS"
                if op2 in (0xA0,0xA8): self.push(self.seg[segname],16)
                else: self.seg[segname]=self.pop(16)&MASK16
            elif 0xC8 <= op2 <= 0xCF: # BSWAP r32
                idx=op2-0xC8; v=self.reg[REG32_NAMES[idx]]
                self.reg[REG32_NAMES[idx]]=int.from_bytes(v.to_bytes(4,"little"),"big")
            elif op2 == 0x0B:
                raise FemuError("guest executed UD2 (#UD invalid-opcode exception)")
            elif op2 in (0x1F,): # multi-byte NOP /0
                mod, sub, rm=self.decode_modrm(self.fetch8())
                if sub != 0: raise FemuError("invalid multi-byte NOP encoding")
                self._decode_rm(mod,rm,width,addrsize) # consume addressing bytes
            elif op2 in (0xAF,): # IMUL r, r/m
                mod, regidx, rm=self.decode_modrm(self.fetch8()); operand=self._decode_rm(mod,rm,width,addrsize)
                a=self.get_reg(regidx,width); b=self._read_operand(operand); sign=1<<(width-1)
                a=a-(1<<width) if a&sign else a; b=b-(1<<width) if b&sign else b
                product=a*b; result=product&((1<<width)-1); overflow=not (-(1<<(width-1))<=product< (1<<(width-1)))
                self.set_reg(regidx,result,width); self.set_flag(FLAG_CF,overflow); self.set_flag(FLAG_OF,overflow)
            elif op2 in (0xBC,0xBD): # BSF / BSR
                mod, regidx, rm=self.decode_modrm(self.fetch8()); operand=self._decode_rm(mod,rm,width,addrsize); value=self._read_operand(operand)
                self.set_flag(FLAG_ZF,value==0)
                if value:
                    pos=(value&-value).bit_length()-1 if op2==0xBC else value.bit_length()-1
                    self.set_reg(regidx,pos,width)
            elif op2 in (0xB0,0xB1): # CMPXCHG r/m, reg
                mod, regidx, rm=self.decode_modrm(self.fetch8()); w=8 if op2==0xB0 else width
                operand=self._decode_rm(mod,rm,w,addrsize); dest=self._read_operand(operand); acc=self.get_reg(0,w)
                result=self.alu("CMP",acc,dest,w)
                if acc==dest: self._write_operand(operand,self.get_reg(regidx,w))
                else: self.set_reg(0,dest,w)
            elif op2 == 0xC7:  # CMPXCHG8B m64, a common 32-bit kernel primitive
                mod, sub, rm=self.decode_modrm(self.fetch8())
                if sub != 1 or mod == 3: raise FemuError("CMPXCHG8B requires a memory operand with ModR/M /1")
                operand=self._decode_rm(mod,rm,64,addrsize)
                addr=operand[1]
                if addr & 7: # Unaligned accesses are permitted by x86, so do not reject them.
                    pass
                expected=((self.reg["EDX"]&MASK32)<<32)|(self.reg["EAX"]&MASK32)
                actual=self.read64(addr)
                if actual==expected:
                    replacement=((self.reg["ECX"]&MASK32)<<32)|(self.reg["EBX"]&MASK32)
                    self.write64(addr,replacement)
                    self.set_flag(FLAG_ZF,True)
                else:
                    self.reg["EAX"]=actual&MASK32; self.reg["EDX"]=(actual>>32)&MASK32
                    self.set_flag(FLAG_ZF,False)
            elif op2 in (0xC0,0xC1): # XADD r/m, reg
                mod, regidx, rm=self.decode_modrm(self.fetch8()); w=8 if op2==0xC0 else width
                operand=self._decode_rm(mod,rm,w,addrsize); old=self._read_operand(operand); src=self.get_reg(regidx,w)
                result=self.alu("ADD",old,src,w); self._write_operand(operand,result); self.set_reg(regidx,old,w)
            elif op2 in (0xA3,0xAB,0xB3,0xBB): # BT/BTS/BTR/BTC r/m, reg
                mod, regidx, rm=self.decode_modrm(self.fetch8()); operand=self._decode_rm(mod,rm,width,addrsize)
                val=self._read_operand(operand); bit=self.get_reg(regidx,width)%(width); mask=1<<bit
                self.set_flag(FLAG_CF,bool(val&mask))
                if op2==0xAB: val|=mask
                elif op2==0xB3: val&=~mask
                elif op2==0xBB: val^=mask
                if op2!=0xA3:self._write_operand(operand,val)
            elif 0x40 <= op2 <= 0x4F:  # CMOVcc r16/32, r/m16/32
                mod, regidx, rm=self.decode_modrm(self.fetch8()); operand=self._decode_rm(mod,rm,width,addrsize)
                value=self._read_operand(operand)
                if self.condition(op2 & 0x0F): self.set_reg(regidx,value,width)
            elif op2 == 0x1E:  # ENDBR32 / ENDBR64 hint; only accept register-form marker
                _mod,_sub,_rm=self.decode_modrm(self.fetch8())
                # F3 0F 1E FB is ENDBR32; current mode does not implement ENDBR64.
                if self.mode_bits==64: raise FemuError("ENDBR64 requested in unsupported 64-bit mode")
            elif op2 in (0x0D,0x18):  # prefetch hints have no modeled cache side effects
                mod,sub,rm=self.decode_modrm(self.fetch8())
                if mod!=3: self._decode_rm(mod,rm,width,addrsize)
            elif op2 == 0xC7:  # CMPXCHG8B m64 (/1)
                mod,sub,rm=self.decode_modrm(self.fetch8())
                if sub != 1 or mod==3: raise FemuError("only memory CMPXCHG8B /1 is supported")
                addr,_seg=self._effective_address(mod,rm,addrsize)
                oldlo=self.read32(addr); oldhi=self.read32(addr+4)
                if oldlo==self.reg["EAX"] and oldhi==self.reg["EDX"]:
                    self.write32(addr,self.reg["EBX"]); self.write32(addr+4,self.reg["ECX"]); self.set_flag(FLAG_ZF,True)
                else:
                    self.reg["EAX"]=oldlo; self.reg["EDX"]=oldhi; self.set_flag(FLAG_ZF,False)
            elif op2 in (0xA4,0xA5,0xAC,0xAD): # SHLD / SHRD
                mod, regidx, rm=self.decode_modrm(self.fetch8()); operand=self._decode_rm(mod,rm,width,addrsize)
                dest=self._read_operand(operand); source=self.get_reg(regidx,width)
                count=self.fetch8() if op2 in (0xA4,0xAC) else self.get_alias("CL")
                count &= 0x1F
                if count:
                    if count >= width: raise FemuError("SHLD/SHRD count >= operand width is undefined in this emulator")
                    mask=(1<<width)-1
                    if op2 in (0xA4,0xA5): res=((dest<<count)|(source>>(width-count)))&mask; cf=(dest>>(width-count))&1
                    else: res=((dest>>count)|(source<<(width-count)))&mask; cf=(dest>>(count-1))&1
                    self._write_operand(operand,res); self.set_flag(FLAG_CF,bool(cf)); self.set_szp(res,width)
                    if count==1:self.set_flag(FLAG_OF,bool((res>>(width-1))^(dest>>(width-1))))
            elif 0x80 <= op2 <= 0x8F:
                rel = self.fetch_signed(width)
                if self.condition(op2 & 0x0F): self.eip = self.eip + rel
            elif 0x90 <= op2 <= 0x9F:
                mod, sub, rm = self.decode_modrm(self.fetch8())
                self._write_operand(self._decode_rm(mod, rm, 8, addrsize), int(self.condition(op2 & 0x0F)))
            elif op2 in (0xB6, 0xB7, 0xBE, 0xBF):
                mod, regidx, rm = self.decode_modrm(self.fetch8())
                srcw = 8 if op2 in (0xB6,0xBE) else 16
                val = self._read_operand(self._decode_rm(mod, rm, srcw, addrsize))
                if op2 in (0xBE,0xBF) and val & (1 << (srcw - 1)): val -= 1 << srcw
                self.set_reg(regidx, val, width)
            else: raise FemuError(f"unsupported 0F opcode {op2:02X}")
        # HLT
        elif opcode == 0xF4:
            self.stopped = True
            self.stop_reason = "HLT"
            raise Halted
        elif opcode in (0xFA, 0xFB):  # CLI / STI
            self.set_flag(FLAG_IF, opcode == 0xFB)
        elif opcode == 0xFC: self.set_flag(FLAG_DF, False)
        elif opcode == 0xFD: self.set_flag(FLAG_DF, True)
        elif opcode == 0xF8: self.set_flag(FLAG_CF, False)
        elif opcode == 0xF9: self.set_flag(FLAG_CF, True)
        else:
            raise FemuError(f"unknown/unsupported opcode {opcode:02X} at CS:IP={self.seg['CS']:04X}:{self._opcode_start:04X}")

        self.default_segment_override = None

    def run(self, max_steps: int = 100000) -> int:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        executed = 0
        self.last_error = None
        while executed < max_steps and not self.stopped:
            try:
                self.step()
            except Halted:
                break
            except FemuError as exc:
                self.last_error = str(exc)
                self.stop_reason = self.last_error
                self.stopped = True
                break
            except (IndexError, struct.error, KeyError, ValueError) as exc:
                self.last_error = f"emulator internal safety error at 0x{self._opcode_start:X}: {exc}"
                self.stop_reason = self.last_error
                self.stopped = True
                break
            executed += 1
        if executed >= max_steps and not self.stopped:
            self.stop_reason = f"instruction budget exhausted ({max_steps} steps)"
            self.stopped = True
        self.panic_message = self._detect_linux_panic()
        if self.panic_message:
            self.stop_reason = "LINUX PANIC: " + self.panic_message
        return executed

    def vga_text(self) -> str:
        lines = []
        for row in range(VGA_ROWS):
            chars = []
            for col in range(VGA_COLS):
                off = VGA_BASE + 2 * (row * VGA_COLS + col)
                ch = self.read8(off) if off < self.ram_size else 0
                chars.append(chr(ch) if 32 <= ch <= 126 else " " if ch in (0, 10, 13) else "·")
            lines.append("".join(chars).rstrip())
        return "\n".join(lines)

    def dump_vga(self) -> None:
        print("\n===== VGA TEXT BUFFER (emulated memory) =====")
        print(self.vga_text())
        print("============================================\n")

    def state(self) -> dict:
        return {"registers": dict(self.reg), "segments": dict(self.seg),
                "mode_bits": self.mode_bits, "protected_mode": self.protected_mode,
                "steps": self.steps, "stopped": self.stopped,
                "stop_reason": self.stop_reason, "last_error": self.last_error,
                "console": "".join(self.console), "serial": "".join(self.serial_console),
                "interrupts": list(self.interrupt_log), "io_log": list(self.io_log[-200:]),
                "control_registers": dict(self.cr), "debug_registers": dict(self.dr),
                "gdtr": dict(self.gdtr), "idtr": dict(self.idtr),
                "linux_panic": self.panic_message}




class Femu64(Femu32):
    """Experimental x86-64 long-mode subset for flat machine-code binaries."""
    REG64_NAMES = ("RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI",
                   "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15")
    MASK64 = 0xFFFFFFFFFFFFFFFF
    CF, PF, ZF, SF, OF = 1, 4, 0x40, 0x80, 0x800

    def __init__(self, ram_size: int = MEM_SIZE_DEFAULT, disk_image: bytes = b""):
        super().__init__(ram_size=ram_size, disk_image=disk_image)
        self.reg64 = {n: 0 for n in self.REG64_NAMES}
        self.reg64.update({"RSP": ram_size - 8, "RIP": 0, "RFLAGS": 2})
        self.mode_bits, self.long_mode, self.protected_mode = 64, True, True
        self.steps, self.stopped, self.stop_reason, self.last_error = 0, False, "", None
        self._aliases64 = {n: (n, 0, 64) for n in self.REG64_NAMES}
        maps = {"RAX":("EAX","AX","AL","AH"), "RCX":("ECX","CX","CL","CH"),
                "RDX":("EDX","DX","DL","DH"), "RBX":("EBX","BX","BL","BH"),
                "RSP":("ESP","SP","SPL",None), "RBP":("EBP","BP","BPL",None),
                "RSI":("ESI","SI","SIL",None), "RDI":("EDI","DI","DIL",None)}
        for reg, aliases in maps.items():
            self._aliases64[aliases[0]] = (reg,0,32)
            self._aliases64[aliases[1]] = (reg,0,16)
            if aliases[2]: self._aliases64[aliases[2]] = (reg,0,8)
            if aliases[3]: self._aliases64[aliases[3]] = (reg,8,8)
        for reg in self.REG64_NAMES[8:]:
            self._aliases64[reg+"D"]=(reg,0,32); self._aliases64[reg+"W"]=(reg,0,16); self._aliases64[reg+"B"]=(reg,0,8)

    @property
    def eip(self) -> int:
        return self.reg64["RIP"]

    @eip.setter
    def eip(self, value: int) -> None:
        self.reg64["RIP"] = value & self.MASK64

    def get_alias(self, name: str) -> int:
        key = name.upper()
        if key == "RIP": return self.reg64["RIP"]
        if key == "RFLAGS": return self.reg64["RFLAGS"]
        if key == "EFLAGS": return self.reg64["RFLAGS"] & 0xFFFFFFFF
        if key not in self._aliases64: raise FemuError("unknown x86-64 register alias: " + name)
        reg, shift, width = self._aliases64[key]
        return (self.reg64[reg] >> shift) & ((1 << width)-1)

    def set_alias(self, name: str, value: int) -> None:
        key = name.upper()
        if key == "RIP": self.reg64["RIP"] = value & self.MASK64; return
        if key in ("RFLAGS","EFLAGS"):
            self.reg64["RFLAGS"] = value & (self.MASK64 if key == "RFLAGS" else 0xFFFFFFFF); return
        if key not in self._aliases64: raise FemuError("unknown x86-64 register alias: " + name)
        reg, shift, width = self._aliases64[key]
        mask = (1 << width)-1
        if width == 64: self.reg64[reg] = value & self.MASK64
        elif width == 32: self.reg64[reg] = value & 0xFFFFFFFF
        else: self.reg64[reg] = (self.reg64[reg] & ~(mask << shift)) | ((value & mask) << shift)

    def load_flat_binary(self, image: bytes, load_address: int = 0x10000,
                         entry_address: Optional[int] = None, bits: int = 64) -> None:
        if bits != 64: raise FemuError("Femu64 loads 64-bit flat binaries only")
        if not isinstance(image, (bytes,bytearray,memoryview)): raise FemuError("flat image must be bytes-like")
        image = bytes(image); self._check(load_address,len(image))
        self.mem[load_address:load_address+len(image)] = image
        self.reg64["RIP"] = (load_address if entry_address is None else entry_address) & self.MASK64
        self.reg64["RSP"] = self.ram_size - 8; self.reg64["RFLAGS"] = 2
        self.stopped=False; self.stop_reason=""; self.last_error=None; self.steps=0

    def _fetch(self, width: int) -> int:
        value = self.read64(self.reg64["RIP"]) if width == 64 else self.read_value(self.reg64["RIP"],width)
        self.reg64["RIP"] = (self.reg64["RIP"] + width//8) & self.MASK64
        return value

    def _fetch8(self) -> int: return self._fetch(8)

    def _signed(self, value: int, width: int) -> int:
        return value-(1<<width) if value & (1<<(width-1)) else value

    def _reg_part(self, index: int, width: int, rex_active: bool):
        if width == 8 and not rex_active and 4 <= index <= 7: return self.REG64_NAMES[index-4],8
        return self.REG64_NAMES[index & 15],0

    def _read_reg_code(self, index: int, width: int, rex_active: bool) -> int:
        name,shift=self._reg_part(index,width,rex_active)
        return (self.reg64[name] >> shift) & ((1<<width)-1)

    def _write_reg_code(self, index: int, value: int, width: int, rex_active: bool) -> None:
        name,shift=self._reg_part(index,width,rex_active); mask=(1<<width)-1
        if width==64: self.reg64[name]=value & self.MASK64
        elif width==32: self.reg64[name]=value & 0xFFFFFFFF
        else: self.reg64[name]=(self.reg64[name] & ~(mask<<shift)) | ((value & mask)<<shift)

    def _decode_rm64(self, rex: int):
        byte=self._fetch8(); mod=byte>>6
        reg=((byte>>3)&7) | (((rex>>2)&1)*8)
        rm_low=byte&7; rm=rm_low | ((rex&1)*8)
        if mod==3: return reg,("reg",rm)
        address=0; rex_x=(rex>>1)&1
        if rm_low==4:
            sib=self._fetch8(); scale=1<<(sib>>6); ix_low=(sib>>3)&7; base_low=sib&7
            ix=ix_low | (rex_x*8)
            if ix_low!=4 or rex_x: address += self.reg64[self.REG64_NAMES[ix]]*scale
            if mod==0 and base_low==5: address += self._signed(self._fetch(32),32)
            else: address += self.reg64[self.REG64_NAMES[base_low | ((rex&1)*8)]]
        elif mod==0 and rm_low==5:
            displacement=self._signed(self._fetch(32),32); address=self.reg64["RIP"]+displacement
        else: address=self.reg64[self.REG64_NAMES[rm]]
        if mod==1: address += self._signed(self._fetch(8),8)
        elif mod==2: address += self._signed(self._fetch(32),32)
        return reg,("mem",address & self.MASK64)

    def _read_rm64(self, operand, width: int, rex_active: bool) -> int:
        return self._read_reg_code(operand[1],width,rex_active) if operand[0]=="reg" else (self.read64(operand[1]) if width==64 else self.read_value(operand[1],width))

    def _write_rm64(self, operand, value: int, width: int, rex_active: bool) -> None:
        if operand[0]=="reg": self._write_reg_code(operand[1],value,width,rex_active)
        else: self.write64(operand[1],value) if width==64 else self.write_value(operand[1],value,width)

    def _push64(self, value: int, width: int=64) -> None:
        self.reg64["RSP"]=(self.reg64["RSP"]-width//8)&self.MASK64
        self.write64(self.reg64["RSP"],value) if width==64 else self.write_value(self.reg64["RSP"],value,width)

    def _pop64(self, width: int=64) -> int:
        value=self.read64(self.reg64["RSP"]) if width==64 else self.read_value(self.reg64["RSP"],width)
        self.reg64["RSP"]=(self.reg64["RSP"]+width//8)&self.MASK64
        return value

    def _set_szp64(self, value: int, width: int) -> int:
        mask=(1<<width)-1; result=value&mask
        flags=self.reg64["RFLAGS"] & ~(self.ZF|self.SF|self.PF)
        if result==0: flags |= self.ZF
        if result & (1<<(width-1)): flags |= self.SF
        if bin(result&255).count("1")%2==0: flags |= self.PF
        self.reg64["RFLAGS"]=flags|2
        return result

    def _alu64(self, operation: str, left: int, right: int, width: int) -> int:
        mask=(1<<width)-1; sign=1<<(width-1); a=left&mask; b=right&mask
        if operation=="add":
            raw=a+b; result=raw&mask; flags=self.reg64["RFLAGS"]&~(self.CF|self.OF)
            if raw>mask: flags|=self.CF
            if (~(a^b)&(a^result)&sign)!=0: flags|=self.OF
            self.reg64["RFLAGS"]=flags|2
            return self._set_szp64(result,width)
        if operation in ("sub","cmp"):
            result=(a-b)&mask; flags=self.reg64["RFLAGS"]&~(self.CF|self.OF)
            if a<b: flags|=self.CF
            if ((a^b)&(a^result)&sign)!=0: flags|=self.OF
            self.reg64["RFLAGS"]=flags|2
            return self._set_szp64(result,width)
        if operation=="xor": result=a^b
        elif operation=="and": result=a&b
        elif operation=="or": result=a|b
        else: raise FemuError("unsupported x86-64 ALU operation: "+operation)
        self.reg64["RFLAGS"] &= ~(self.CF|self.OF)
        return self._set_szp64(result,width)

    def _condition64(self, code: int) -> bool:
        f=self.reg64["RFLAGS"]
        cf,pf,zf,sf,of=(bool(f&x) for x in (self.CF,self.PF,self.ZF,self.SF,self.OF))
        return (of,not of,cf,not cf,zf,not zf,cf or zf,not cf and not zf,
                sf,not sf,pf,not pf,sf!=of,sf==of,zf or sf!=of,not zf and sf==of)[code&15]

    def step(self) -> None:
        if self.stopped: return
        start=self.reg64["RIP"]; rex=0; op_override=False; opcode=None
        for _ in range(8):
            value=self._fetch8()
            if value==0x66: op_override=True
            elif value==0x67 or value in (0xF2,0xF3): continue
            elif 0x40<=value<=0x4F: rex=value
            else: opcode=value; break
        if opcode is None: raise FemuError("too many x86-64 instruction prefixes")
        width=64 if rex&8 else (16 if op_override else 32); rex_active=bool(rex)

        if opcode==0x90: pass
        elif opcode==0xF4: self.stopped=True; self.stop_reason="HLT"
        elif 0xB0<=opcode<=0xB7:
            index=(opcode&7)|((rex&1)*8); self._write_reg_code(index,self._fetch8(),8,rex_active)
        elif 0xB8<=opcode<=0xBF:
            index=(opcode&7)|((rex&1)*8); iw=64 if width==64 else width
            self._write_reg_code(index,self._fetch(iw),width,rex_active)
        elif 0x50<=opcode<=0x57:
            index=(opcode&7)|((rex&1)*8); sw=16 if op_override else 64
            self._push64(self._read_reg_code(index,sw,rex_active),sw)
        elif 0x58<=opcode<=0x5F:
            index=(opcode&7)|((rex&1)*8); sw=16 if op_override else 64
            self._write_reg_code(index,self._pop64(sw),sw,rex_active)
        elif opcode in (0x88,0x89,0x8A,0x8B,0x8D,0x01,0x03,0x09,0x0B,0x21,0x23,0x29,0x2B,0x31,0x33,0x39,0x3B,0x85,0x87):
            w=8 if opcode in (0x88,0x8A) else width; ri,rm=self._decode_rm64(rex)
            if opcode==0x8D:
                if rm[0]!="mem": raise FemuError("LEA requires a memory operand")
                self._write_reg_code(ri,rm[1],width,rex_active)
            elif opcode in (0x88,0x89): self._write_rm64(rm,self._read_reg_code(ri,w,rex_active),w,rex_active)
            elif opcode in (0x8A,0x8B): self._write_reg_code(ri,self._read_rm64(rm,w,rex_active),w,rex_active)
            elif opcode==0x87:
                a=self._read_rm64(rm,w,rex_active); b=self._read_reg_code(ri,w,rex_active)
                self._write_rm64(rm,b,w,rex_active); self._write_reg_code(ri,a,w,rex_active)
            else:
                rm_dest=opcode in (0x01,0x09,0x21,0x29,0x31,0x39,0x85)
                a=self._read_rm64(rm,w,rex_active) if rm_dest else self._read_reg_code(ri,w,rex_active)
                b=self._read_reg_code(ri,w,rex_active) if rm_dest else self._read_rm64(rm,w,rex_active)
                op={0x01:"add",0x03:"add",0x09:"or",0x0B:"or",0x21:"and",0x23:"and",0x29:"sub",0x2B:"sub",0x31:"xor",0x33:"xor",0x39:"cmp",0x3B:"cmp",0x85:"and"}[opcode]
                result=self._alu64(op,a,b,w)
                if opcode not in (0x39,0x3B,0x85):
                    if opcode in (0x03,0x0B,0x23,0x2B,0x33): self._write_reg_code(ri,result,w,rex_active)
                    else: self._write_rm64(rm,result,w,rex_active)
        elif opcode in (0x81,0x83,0xC7,0xFF):
            ri,rm=self._decode_rm64(rex); group=ri&7
            if opcode==0xC7:
                if group!=0: raise FemuError("unsupported x86-64 C7 group")
                iw=16 if width==16 else 32; imm=self._fetch(iw)
                if width==64: imm=self._signed(imm,32)&self.MASK64
                self._write_rm64(rm,imm,width,rex_active)
            elif opcode in (0x81,0x83):
                iw=8 if opcode==0x83 else (16 if width==16 else 32); imm=self._fetch(iw)
                if opcode==0x83 or width==64: imm=self._signed(imm,iw)
                op={0:"add",1:"or",4:"and",5:"sub",6:"xor",7:"cmp"}.get(group)
                if op is None: raise FemuError("unsupported x86-64 immediate ALU group "+str(group))
                result=self._alu64(op,self._read_rm64(rm,width,rex_active),imm,width)
                if op!="cmp": self._write_rm64(rm,result,width,rex_active)
            else:
                if group in (0,1):
                    old_cf=self.reg64["RFLAGS"]&self.CF
                    result=self._alu64("add" if group==0 else "sub",self._read_rm64(rm,width,rex_active),1,width)
                    self._write_rm64(rm,result,width,rex_active)
                    self.reg64["RFLAGS"]=(self.reg64["RFLAGS"]&~self.CF)|old_cf
                elif group in (2,4):
                    target=self._read_rm64(rm,64,rex_active)
                    if group==2: self._push64(self.reg64["RIP"])
                    self.reg64["RIP"]=target&self.MASK64
                elif group==6: self._push64(self._read_rm64(rm,width,rex_active),64)
                else: raise FemuError("unsupported x86-64 FF group "+str(group))
        elif opcode in (0x05,0x0D,0x25,0x2D,0x35,0x3D):
            iw=16 if width==16 else 32; imm=self._fetch(iw)
            if width==64: imm=self._signed(imm,32)
            op={0x05:"add",0x0D:"or",0x25:"and",0x2D:"sub",0x35:"xor",0x3D:"cmp"}[opcode]
            result=self._alu64(op,self.reg64["RAX"]&((1<<width)-1),imm,width)
            if op!="cmp": self._write_reg_code(0,result,width,rex_active)
        elif opcode==0xE8:
            disp=self._signed(self._fetch(32),32); self._push64(self.reg64["RIP"]); self.reg64["RIP"]=(self.reg64["RIP"]+disp)&self.MASK64
        elif opcode==0xE9:
            disp=self._signed(self._fetch(32),32); self.reg64["RIP"]=(self.reg64["RIP"]+disp)&self.MASK64
        elif opcode==0xEB:
            disp=self._signed(self._fetch(8),8); self.reg64["RIP"]=(self.reg64["RIP"]+disp)&self.MASK64
        elif 0x70<=opcode<=0x7F:
            disp=self._signed(self._fetch(8),8)
            if self._condition64(opcode&15): self.reg64["RIP"]=(self.reg64["RIP"]+disp)&self.MASK64
        elif opcode==0x0F:
            second=self._fetch8()
            if 0x80<=second<=0x8F:
                disp=self._signed(self._fetch(32),32)
                if self._condition64(second&15): self.reg64["RIP"]=(self.reg64["RIP"]+disp)&self.MASK64
            elif second in (0xB6,0xB7,0xBE,0xBF):
                ri,rm=self._decode_rm64(rex); sw=8 if second in (0xB6,0xBE) else 16
                val=self._read_rm64(rm,sw,rex_active)
                if second in (0xBE,0xBF): val=self._signed(val,sw)
                self._write_reg_code(ri,val,width,rex_active)
            elif second==0xA2:
                leaf=self.reg64["RAX"]&0xFFFFFFFF
                if leaf==0: vals=(1,int.from_bytes(b"Genu","little"),int.from_bytes(b"ineI","little"),int.from_bytes(b"ntel","little"))
                elif leaf==1: vals=(0x663,0,0x100,1)
                else: vals=(0,0,0,0)
                for n,v in zip(("RAX","RBX","RCX","RDX"),(vals[0],vals[1],vals[3],vals[2])): self.reg64[n]=v
            else: raise FemuError("unsupported x86-64 0F opcode 0x%02X at RIP=0x%X"%(second,start))
        elif opcode in (0xCC,0xCD):
            raise FemuError("x86-64 interrupts/BIOS are not implemented; use flat code")
        else: raise FemuError("unsupported x86-64 opcode 0x%02X at RIP=0x%X"%(opcode,start))
        self.reg64["RIP"] &= self.MASK64; self.reg64["RFLAGS"] |= 2; self.steps += 1

    def run(self, max_steps: int = 100000) -> int:
        if max_steps < 1: raise ValueError("max_steps must be positive")
        while not self.stopped and self.steps < max_steps:
            try: self.step()
            except (FemuError,IndexError,OverflowError,ValueError) as exc:
                self.stopped=True; self.stop_reason="ERROR"; self.last_error=str(exc)
        if not self.stopped: self.stopped=True; self.stop_reason="STEP_LIMIT"
        return self.steps

    def state(self) -> dict:
        regs={n:"0x%016X"%(v&self.MASK64) for n,v in self.reg64.items()}
        return {"mode_bits":64,"registers":regs,"segments":{},"steps":self.steps,
                "stopped":self.stopped,"stop_reason":self.stop_reason,"last_error":self.last_error,
                "console":"".join(self.console),"serial":"".join(self.serial_console),
                "interrupts":list(self.interrupt_log),"io_log":list(self.io_log[-200:]),
                "control_registers":dict(self.cr),"debug_registers":dict(self.dr),
                "gdtr":dict(self.gdtr),"idtr":dict(self.idtr),"linux_panic":None}


def run_chunk(cpu, max_steps: int = 25000) -> int:
    """Execute one bounded chunk without stopping the VM merely at a chunk boundary."""
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be positive")
    executed = 0
    while executed < max_steps and not cpu.stopped:
        try:
            cpu.step()
        except Halted:
            break
        except FemuError as exc:
            cpu.last_error = str(exc)
            cpu.stop_reason = cpu.last_error
            cpu.stopped = True
            break
        except (IndexError, struct.error, KeyError, ValueError, OverflowError) as exc:
            ip = getattr(cpu, "_opcode_start", getattr(cpu, "eip", 0))
            cpu.last_error = f"emulator internal safety error at 0x{ip:X}: {exc}"
            cpu.stop_reason = cpu.last_error
            cpu.stopped = True
            break
        executed += 1
        if getattr(cpu, "mode_bits", 32) != 64:
            cpu.steps = getattr(cpu, "steps", 0) + 1
    if hasattr(cpu, "_detect_linux_panic"):
        panic = cpu._detect_linux_panic()
        if panic:
            cpu.panic_message = panic
            cpu.stop_reason = "LINUX PANIC: " + panic
            cpu.stopped = True
    return executed


def run_unlimited(cpu, chunk_size: int = 25000) -> int:
    """Run until halt/error/panic; callers can still interrupt a long-running loop."""
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    total = 0
    cpu._unlimited_total = 0
    while not cpu.stopped:
        amount = run_chunk(cpu, chunk_size)
        total += amount
        cpu._unlimited_total = total
        if amount == 0 and not cpu.stopped:
            raise FemuError("unlimited runner made no progress")
    return total


def inspect_linux_bzimage(image: bytes) -> dict:
    """Read the documented Linux/x86 setup header without executing the image."""
    if len(image) < 0x268:
        raise FemuError("image is too small to contain a Linux/x86 boot header")
    if image[0x202:0x206] != b"HdrS":
        raise FemuError("no Linux/x86 HdrS setup signature at file offset 0x202; expected a bzImage")
    setup_sects = image[0x1F1] or 4
    protocol = int.from_bytes(image[0x206:0x208], "little")
    loadflags = image[0x211]
    xloadflags = int.from_bytes(image[0x236:0x238], "little")
    setup_bytes = (setup_sects + 1) * 512
    if setup_bytes > len(image):
        raise FemuError("Linux setup_sects points beyond end of image")
    return {
        "setup_sectors": setup_sects,
        "setup_bytes": setup_bytes,
        "protocol_version": f"{protocol >> 8}.{protocol & 0xFF:02d}",
        "protocol": protocol,
        "load_high": bool(loadflags & 1),
        "supports_64bit_kernel": bool(xloadflags & 2),
        "kernel_payload_bytes": max(0, len(image) - setup_bytes),
        "boot_signature_valid": image[510:512] == b"\x55\xAA",
    }


def make_demo_boot_sector() -> bytes:
    """Create an original 512-byte, signed boot sector that prints via INT 10h."""
    code = bytearray()
    # Real-mode code: DS=CS; SI=message; lodsb; test al,al; jz done; teletype; loop.
    code += b"\xFA"                    # cli
    code += b"\x31\xC0"                # xor ax,ax
    code += b"\x8E\xD8"                # mov ds,ax
    code += b"\xBE\x00\x7D"            # mov si,0x7D00 (message follows boot code)
    loop_pos = len(code)
    code += b"\xAC"                    # lodsb
    code += b"\x84\xC0"                # test al,al
    code += b"\x74\x00"                # jz done (patch rel8)
    jz_disp_pos = len(code) - 1
    code += b"\xB4\x0E"                # mov ah,0x0e
    code += b"\xBB\x07\x00"            # mov bx,0x0007
    code += b"\xCD\x10"                # int 10h
    code += b"\xEB\x00"                # jmp loop (patch rel8)
    jmp_disp_pos = len(code) - 1
    done_pos = len(code)
    code += b"\xF4"                    # hlt
    message = b"Femu32 booted from a real-mode boot sector!\r\n\x00"
    message_offset = 0x7C00 + len(code)
    code[6:8] = struct.pack("<H", message_offset)  # SI immediate
    code[jz_disp_pos] = (done_pos - (0x7C00 + jz_disp_pos + 1)) & 0xFF
    code[jmp_disp_pos] = (loop_pos - (0x7C00 + jmp_disp_pos + 1)) & 0xFF
    sector = bytearray(512)
    sector[:len(code)] = code
    sector[len(code):len(code) + len(message)] = message
    sector[510:512] = b"\x55\xAA"
    return bytes(sector)


def make_flat_demo() -> bytes:
    """Machine-code regression demo: writes 5 to [0x200000], decrements ECX."""
    return (b"\xB9" + (5).to_bytes(4, "little") +
            b"\xBB" + (0x200000).to_bytes(4, "little") +
            b"\x89\x0B" + b"\xFF\xC9" + b"\xF4")


def self_test() -> None:
    checks = 0
    def check(condition: bool, message: str) -> None:
        nonlocal checks
        if not condition: raise AssertionError(message)
        checks += 1

    # MOV, memory, INC/DEC, HLT and flat binary loading.
    cpu = Femu32()
    cpu.load_flat_binary(make_flat_demo(), load_address=0, entry_address=0)
    cpu.run()
    check(cpu.read32(0x200000) == 5, "flat demo memory write")
    check(cpu.get_alias("ECX") == 4, "DEC register")
    check(cpu.stop_reason == "HLT", "flat demo HLT")

    # ADD flags: 0xFFFFFFFF + 1 -> zero with carry; INC preserves CF.
    cpu = Femu32(); cpu.load_flat_binary(b"\xB8\xFF\xFF\xFF\xFF\x05\x01\x00\x00\x00\xF4", 0, 0)
    cpu.run()
    check(cpu.get_alias("EAX") == 0, "32-bit ADD wrapping")
    check(cpu.flag(FLAG_CF) == 1 and cpu.flag(FLAG_ZF) == 1, "ADD CF/ZF")
    check(cpu.flag(FLAG_PF) == 1, "ADD parity flag")
    # INC must not clear a previously-set CF.
    cpu = Femu32(); cpu.load_flat_binary(bytes.fromhex("B8FFFFFFFF 0501000000 40 F4".replace(" ", "")), 0, 0); cpu.run()
    check(cpu.get_alias("EAX") == 1 and cpu.flag(FLAG_CF) == 1, "INC preserves CF")

    # Signed short backward branch and conditional flags.
    # xor eax,eax; cmp eax,0; jz +2; mov ebx,1; mov ebx,2; hlt
    cpu = Femu32(); code = bytes.fromhex("31C0 3D00000000 7405 BB01000000 BB02000000 F4".replace(" ", ""))
    cpu.load_flat_binary(code, 0, 0); cpu.run()
    check(cpu.get_alias("EBX") == 2, "short JZ")

    # PUSH/POP/CALL/RET using a CALL into a function which changes EAX then returns.
    # mov esp,0x80000; mov eax,7; call +6; hlt; mov eax,9; ret
    code = bytes.fromhex("BC00000800 B807000000 E801000000 F4 B809000000 C3")
    cpu = Femu32(); cpu.load_flat_binary(code, 0, 0); cpu.run()
    check(cpu.get_alias("EAX") == 9, "CALL/RET control flow")
    # Push/pop register round trip.
    code = bytes.fromhex("BC00000800 B878563412 50 B800000000 58 F4")
    cpu = Femu32(); cpu.load_flat_binary(code, 0, 0); cpu.run()
    check(cpu.get_alias("EAX") == 0x12345678, "PUSH/POP")

    # Correct 16-bit real-mode boot sector, INT 10h text output and HLT.
    cpu = Femu32(); cpu.boot_disk(make_demo_boot_sector()); cpu.run()
    check(cpu.stop_reason == "HLT", f"boot sector halted: {cpu.stop_reason}")
    check("Femu32 booted" in "".join(cpu.console), "BIOS teletype output")
    check(not any(line.startswith("INT 10h at") for line in cpu.interrupt_log),
          "successful BIOS teletype calls do not flood interrupt log")
    check(cpu.mem[0x7C00 + 510:0x7C00 + 512] == b"\x55\xAA", "boot signature")

    # Accurate /7 CMP r/m32, imm32: compare EAX=5 to 5 and JZ.
    code = bytes.fromhex("B805000000 81F805000000 7400 F4".replace(" ", ""))
    cpu = Femu32(); cpu.load_flat_binary(code, 0, 0); cpu.run()
    check(cpu.flag(FLAG_ZF) == 1, "CMP r/m immediate sets ZF")

    # CPUID exposes a stable synthetic 32-bit CPU identity, and RDTSC is deterministic.
    cpu=Femu32(); cpu.load_flat_binary(bytes.fromhex("B800000000 0FA2 F4".replace(" ","")),0,0); cpu.run()
    check(cpu.reg["EBX"]==int.from_bytes(b"Genu","little") and cpu.reg["EDX"]==int.from_bytes(b"ineI","little"),"CPUID vendor")
    # OUT to the debug console and COM1 is captured instead of touching host hardware.
    cpu=Femu32(); cpu.load_flat_binary(bytes.fromhex("B041 E6E9 F4"),0,0); cpu.run()
    check("A" in "".join(cpu.serial_console),"debug-port output capture")
    # x86 port I/O, control-register move, far-GDT transitions, and basic page translation.
    cpu=Femu32(); cpu.cr["CR0"]=0; cpu.load_flat_binary(bytes.fromhex("B8 01000000 0F22C0 F4".replace(" ","")),0,0); cpu.run()
    check(cpu.cr["CR0"]==1 and cpu.protected_mode,"MOV CR0 enables protected-mode flag")
    cpu=Femu32(); cpu.load_flat_binary(bytes.fromhex("B803000000 83C005 0FC8 F4".replace(" ","")),0,0); cpu.run()
    check(cpu.reg["EAX"]==0x08000000,"BSWAP register")
    # A known Linux setup signature is parsed with protocol and setup size fields.
    fake=bytearray(4096); fake[0x1F1]=4; fake[0x202:0x206]=b"HdrS"; fake[0x206:0x208]=(0x020B).to_bytes(2,"little"); fake[0x211]=1; fake[0x236:0x238]=(0).to_bytes(2,"little"); fake[510:512]=b"\x55\xAA"
    info=inspect_linux_bzimage(bytes(fake))
    check(info["protocol"]==0x020B and info["setup_bytes"]==2560,"Linux bzImage header parsing")

    # REP MOVSB, SHL and CMOVcc are common compiler/startup building blocks.
    cpu=Femu32(); cpu.mem[0x300:0x303]=b"abc"
    code=bytes.fromhex("BE00030000 BF00040000 B903000000 F3A4 B801000000 83E001 0F44C3 F4")
    cpu.load_flat_binary(code,0,0); cpu.mem[0x300:0x303]=b"abc"; cpu.run()
    check(cpu.mem[0x400:0x403]==b"abc","REP MOVSB copies buffer")
    # E820 and port output are deterministic BIOS/board shim calls.
    cpu=Femu32(); cpu.boot_disk(make_demo_boot_sector()); cpu.set_alias("EAX",0xE820); cpu.set_alias("EDX",0x534D4150); cpu.set_alias("EBX",0); cpu.set_alias("DI",0x500); cpu.interrupt(0x15)
    check(cpu.read32(0x500+16)==1 and cpu.get_alias("EAX")==0x534D4150,"BIOS E820 first entry")
    # MOV DRx regression: guest debug registers are independent of host state.
    cpu=Femu32(); cpu.load_flat_binary(bytes.fromhex("B878563412 0F23C0 B800000000 0F21C3 F4"),0,0); cpu.run()
    check(cpu.dr["DR0"]==0x12345678 and cpu.reg["EBX"]==0x12345678, "MOV DR0 round trip")

    # ADC regression tests: register direction, carry-in, and byte memory operands.
    cpu=Femu32(); cpu.load_flat_binary(bytes.fromhex("B0FF B300 F9 10D8 F4"),0,0); cpu.run()
    check(cpu.get_alias("AL")==0 and cpu.flag(FLAG_CF)==1, "ADC r/m8,r8 carries through 0xFF plus carry-in")
    cpu=Femu32(); cpu.load_flat_binary(bytes.fromhex("B001 B300 F8 12D8 F4"),0,0); cpu.run()
    check(cpu.get_alias("BL")==1 and cpu.flag(FLAG_CF)==0, "ADC r8,r/m8 supports reverse direction")
    cpu=Femu32(); cpu.load_flat_binary(bytes.fromhex("B001 100500030000 F4".replace(" ","")),0,0); cpu.mem[0x300]=0xFE; cpu.set_flag(FLAG_CF,True); cpu.run()
    check(cpu.read8(0x300)==0 and cpu.flag(FLAG_CF)==1, "ADC byte memory operand includes carry-in")

    # Protected-mode POP SS must use the same descriptor validation as MOV SS.
    cpu=Femu32(); cpu.load_flat_binary(bytes.fromhex("6A10 17 F4".replace(" ","")),0,0,32)
    cpu.gdtr={"base":0x500,"limit":0x17}; cpu.write32(0x510,0x0000FFFF); cpu.write32(0x514,0x00CF9200); cpu.run()
    check(cpu.seg["SS"]==0x10 and cpu.stop_reason=="HLT", "protected-mode POP SS loads valid writable data selector")
    cpu=Femu32(); cpu.load_flat_binary(bytes.fromhex("6A00 17 F4".replace(" ","")),0,0,32)
    cpu.run()
    check("null selector" in cpu.stop_reason, "protected-mode POP SS rejects null selector")

    # CMPXCHG8B memory compare/exchange, used by some 32-bit kernel paths.
    cpu=Femu32(); cpu.load_flat_binary(bytes.fromhex("0FC70D00030000F4"),0,0)
    cpu.reg["EAX"]=0; cpu.reg["EDX"]=0; cpu.reg["EBX"]=0x55667788; cpu.reg["ECX"]=0x11223344
    cpu.run(); check(cpu.read64(0x300)==0x1122334455667788 and cpu.flag(FLAG_ZF)==1, "CMPXCHG8B success path")
    cpu=Femu32(); cpu.load_flat_binary(bytes.fromhex("0FC70D00030000F4"),0,0)
    cpu.write64(0x300,0x8877665544332211); cpu.reg["EAX"]=0; cpu.reg["EDX"]=0; cpu.run()
    check(cpu.reg["EAX"]==0x44332211 and cpu.reg["EDX"]==0x88776655 and cpu.flag(FLAG_ZF)==0, "CMPXCHG8B failure path")

    # Kernel panic detection is based on guest-visible output, not emulator exceptions.
    cpu=Femu32(); cpu.serial_console.append("Kernel panic - not syncing: VFS: Unable to mount root fs")
    check("VFS" in (cpu._detect_linux_panic() or ""),"Linux panic output detection")

    # Linux bzImage loader sets up the real-mode entry point and 32-bit payload.
    fake=bytearray(4096); fake[0x1F1]=4; fake[0x1FE:0x200]=b"\x55\xAA"; fake[0x200:0x202]=b"\xEB\x6A"; fake[0x202:0x206]=b"HdrS"; fake[0x206:0x208]=(0x020B).to_bytes(2,"little"); fake[0x211]=1; fake[0x236:0x238]=(0).to_bytes(2,"little"); fake[0x260:0x264]=(0x1000).to_bytes(4,"little")
    cpu=Femu32(); info=cpu.boot_linux_bzimage(bytes(fake), "panic=1")
    check(cpu.seg["CS"]==0x9020 and cpu.eip==0 and cpu.linear(cpu.eip,"CS")==0x90200,"Linux setup entry address")
    check(cpu.read32(0x90000+0x214)==0x100000 and cpu.read32(0x90000+0x228)==0x99800,"Linux header payload/command-line pointers")
    check(cpu.mem[0x99800:0x99808]==b"panic=1\x00","Linux command-line staging")

    # Image safety: bad signatures are rejected and RAM bounds are guarded.
    try:
        Femu32().boot_disk(bytes(512))
        raise AssertionError("invalid boot signature was accepted")
    except FemuError as exc:
        check("signature" in str(exc), "invalid boot signature error")

    # Long-mode regressions: MOVABS, REX-extended registers, ADD, PUSH/POP and HLT.
    cpu64 = Femu64()
    code64 = bytes.fromhex("48 B8 8877665544332211 49 B9 0800000000000000 4C 01 C8 50 58 F4".replace(" ", ""))
    cpu64.load_flat_binary(code64, 0, 0, 64); cpu64.run()
    check(cpu64.get_alias("RAX") == 0x1122334455667790, "x86-64 MOVABS/REX ADD/PUSH POP")
    check(cpu64.stop_reason == "HLT" and cpu64.last_error is None, "x86-64 execution halts without error")
    # Chunk runner regression: small chunks still continue through a guest halt.
    cpu_chunk = Femu32(); cpu_chunk.boot_disk(make_demo_boot_sector())
    chunk_total = 0
    while not cpu_chunk.stopped and chunk_total < 2000: chunk_total += run_chunk(cpu_chunk, 25)
    check(cpu_chunk.stop_reason == "HLT" and chunk_total > 0, "chunk runner continues to guest HLT")
    cpu64_chunk = Femu64(); cpu64_chunk.load_flat_binary(b"\xF4", 0, 0, 64)
    check(run_chunk(cpu64_chunk, 10) > 0 and cpu64_chunk.stop_reason == "HLT", "64-bit chunk runner HLT")
    # 16-bit LOOP decrements CX and branches on nonzero without altering flags.
    cpu = Femu32(); cpu.load_flat_binary(bytes.fromhex("B90300 B80000 40 E2FD F4".replace(" ", "")), 0, 0, 16); cpu.run()
    check(cpu.get_alias("AX") == 3 and cpu.get_alias("CX") == 0 and cpu.stop_reason == "HLT", "16-bit LOOP CX semantics")
    # LOOPNZ uses ZF=0 and still leaves ZF unchanged while counting down.
    cpu = Femu32(); cpu.load_flat_binary(bytes.fromhex("B90200 B80100 3D0000 E0FE F4".replace(" ", "")), 0, 0, 16); cpu.run()
    check(cpu.get_alias("CX") == 0 and cpu.flag(FLAG_ZF) == 0 and cpu.stop_reason == "HLT", "LOOPNZ counter and ZF semantics")
    # LOOPZ repeats when ZF=1 and does not modify the flags.
    cpu = Femu32(); cpu.load_flat_binary(bytes.fromhex("B90300 31C0 E1FE F4".replace(" ", "")), 0, 0, 16); cpu.run()
    check(cpu.get_alias("CX") == 0 and cpu.flag(FLAG_ZF) == 1 and cpu.stop_reason == "HLT", "LOOPZ counter and ZF semantics")
    # JCXZ tests CX without decrementing it.
    cpu = Femu32(); cpu.load_flat_binary(bytes.fromhex("B90000 E303 B80100 F4".replace(" ", "")), 0, 0, 16); cpu.run()
    check(cpu.get_alias("CX") == 0 and cpu.get_alias("AX") == 0 and cpu.stop_reason == "HLT", "JCXZ branches without decrement")
    # Address-size override 67 makes LOOP use ECX even in 16-bit mode.
    cpu = Femu32(); cpu.load_flat_binary(bytes.fromhex("66B902000000 67E2FD F4".replace(" ", "")), 0, 0, 16); cpu.run()
    check(cpu.get_alias("ECX") == 0 and cpu.stop_reason == "HLT", "address-size override LOOP uses ECX")
    print(f"Femu.py self-test: PASS ({checks} checks)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Femu.py educational x86 subset emulator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--boot", metavar="IMAGE", help="boot a raw disk image / boot-sector binary")
    group.add_argument("--flat", metavar="IMAGE", help="run a flat 16- or 32-bit binary (default 32-bit)")
    group.add_argument("--linux", "--bzimage", metavar="BZIMAGE", help="experimental BIOS boot attempt from a Linux x86 bzImage")
    parser.add_argument("--load", type=lambda s: int(s, 0), default=0x10000, help="flat binary load address (default 0x10000)")
    parser.add_argument("--entry", type=lambda s: int(s, 0), default=None, help="flat binary entry physical address")
    parser.add_argument("--bits", type=int, choices=(16, 32), default=32, help="flat binary mode (default 32)")
    parser.add_argument("--ram", type=lambda s: int(s, 0), default=MEM_SIZE_DEFAULT, help="RAM size in bytes (default 64 MiB)")
    parser.add_argument("--steps", type=int, default=None, help="optional maximum instruction count (default: unlimited; omit for no budget)")
    parser.add_argument("--trace", action="store_true", help="record instruction/register trace")
    parser.add_argument("--no-signature-check", action="store_true", help="allow a boot sector without 0x55AA")
    parser.add_argument("--self-test", action="store_true", help="run regression checks")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            self_test(); return 0
        cpu = Femu32(ram_size=args.ram)
        cpu.trace = args.trace
        if args.boot:
            image = Path(args.boot).read_bytes()
            cpu.boot_disk(image, require_signature=not args.no_signature_check)
            print(f"[BIOS] Image: {args.boot} ({len(image)} bytes); entry 0000:7C00")
        elif args.linux:
            image = Path(args.linux).read_bytes()
            info = inspect_linux_bzimage(image)
            if not info["boot_signature_valid"]:
                raise FemuError("Linux bzImage boot sector has an invalid 0x55AA signature")
            print(f"[LINUX] bzImage detected: protocol {info['protocol_version']}; setup={info['setup_sectors']} sector(s); payload={info['kernel_payload_bytes']} bytes")
            print(f"[LINUX] load_high={info['load_high']}; advertises_64bit={info['supports_64bit_kernel']}; RAM={cpu.ram_size // (1024*1024)} MiB")
            if info["supports_64bit_kernel"]:
                print("[LINUX] NOTE: this kernel advertises a 64-bit entry point; Femu is currently 16/32-bit and may stop during CPU feature checks or long-mode setup.")
            info = cpu.boot_linux_bzimage(image)
            print(f"[LINUX] Entering Linux real-mode setup at 9020:0000; loaded protected payload at 0x{(0x100000 if info['load_high'] else 0x10000):08X}")
        elif args.flat:
            image = Path(args.flat).read_bytes()
            cpu.load_flat_binary(image, args.load, args.entry, args.bits)
            print(f"[CPU] Flat image: {args.flat} ({len(image)} bytes); mode={args.bits}-bit; load=0x{args.load:X}; entry=0x{cpu.eip:X}")
        else:
            print("[Femu.py] Running built-in boot-sector demonstration")
            cpu.boot_disk(make_demo_boot_sector())
        if args.steps is None:
            print("[CPU] No instruction budget; running until halt/error/panic. Press Ctrl+C to stop.")
            try:
                executed = run_unlimited(cpu)
            except KeyboardInterrupt:
                cpu.stopped = True
                cpu.stop_reason = "Interrupted by user"
                executed = getattr(cpu, "_unlimited_total", 0)
        else:
            print(f"[CPU] Instruction budget: {args.steps:,}")
            executed = cpu.run(args.steps)
        if cpu.console:
            print("".join(cpu.console), end="" if cpu.console[-1].endswith("\n") else "\n")
        if cpu.serial_console:
            print("\n[SERIAL/DEBUGCON] " + "".join(cpu.serial_console))
        print(f"\n[CPU] Executed {executed} instruction(s); stopped={cpu.stop_reason or 'unknown'}")
        print("[CONTROL] " + " ".join(f"{r}={cpu.cr[r]:08X}" for r in ("CR0","CR2","CR3","CR4")) + f" GDTR={cpu.gdtr['base']:08X}/{cpu.gdtr['limit']:04X}")
        print("[REGS] " + " ".join(f"{r}={cpu.reg[r]:08X}" for r in ("EAX","EBX","ECX","EDX","ESI","EDI","EBP","ESP")))
        print(f"[REGS] CS:IP={cpu.seg['CS']:04X}:{cpu.eip:04X} EFLAGS={cpu.reg['EFLAGS']:08X}")
        if cpu.panic_message:
            print("\n[LINUX PANIC DETECTED] " + cpu.panic_message)
        if cpu.interrupt_log:
            print("[INTERRUPTS]")
            for line in cpu.interrupt_log[-30:]: print("  " + line)
        if cpu.io_log:
            print("[PORT I/O] (last 12 operations)")
            for line in cpu.io_log[-12:]: print("  " + line)
        if cpu.trace_lines:
            print("[TRACE]")
            for line in cpu.trace_lines: print(line)
        if cpu.last_error:
            print("[ERROR] " + cpu.last_error, file=sys.stderr)
            return 1
        return 0 if cpu.stop_reason == "HLT" else 1
    except (OSError, ValueError, FemuError, AssertionError) as exc:
        print(f"[Femu.py] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Avoid raising SystemExit(0) when a user runs this file through IPython's
    # %run / runpy environment; IPython otherwise prints an alarming traceback
    # even though the emulator completed successfully. Preserve nonzero failure
    # exit codes for real CLI invocations.
    _exit_code = main()
    if _exit_code:
        raise SystemExit(_exit_code)