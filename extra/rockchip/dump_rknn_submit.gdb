set pagination off
set breakpoint pending on

python
import gdb, subprocess

SUBMIT_NR = 0x41
DRM_TYPE = ord('d')
SUBMIT_SIZE = 104
TASK_SIZE = 40
TARGETS = {0x81:"PC", 0x201:"CNA", 0x801:"CORE", 0x1001:"DPU", 0x2001:"DPU_RDMA", 0x4001:"PPU", 0x8001:"PPU_RDMA"}
BASE_REGS = {0x4020, 0x500c, 0x5020, 0x502c, 0x5038, 0x600c, 0x700c, 0x7040}

def u32(data, offset): return int.from_bytes(data[offset:offset+4], "little")
def u64(data, offset): return int.from_bytes(data[offset:offset+8], "little")

def mappings():
  rows = []
  for line in gdb.execute("info proc mappings", to_string=True).splitlines():
    fields = line.split()
    if len(fields) >= 5 and fields[0].startswith("0x") and fields[1].startswith("0x"):
      rows.append((int(fields[0],16),int(fields[1],16)))
  return rows

def find_tasks(inferior, count):
  for start,end in mappings():
    if end-start > 1<<20: continue
    try: data = inferior.read_memory(start,min(end-start,1<<16)).tobytes()
    except gdb.MemoryError: continue
    for offset in range(0,len(data)-count*TASK_SIZE+1,8):
      valid, base_command = True, None
      for index in range(count):
        task = data[offset+index*TASK_SIZE:offset+(index+1)*TASK_SIZE]
        flags, op, enable, clear, amount, command_offset, command_address = \
          u32(task,0),u32(task,4),u32(task,8),u32(task,16),u32(task,24),u32(task,28),u64(task,32)
        if flags > 0xffff or op > 0xffff or not enable or enable > 0xffff or not clear or clear > 0xffffff or \
           not 4 <= amount <= 1024 or command_offset&7 or command_address&0xf:
          valid = False
          break
        if base_command is None: base_command = command_address
        elif command_address != base_command: valid = False; break
      if valid: return start+offset, data[offset:offset+count*TASK_SIZE]
  return None,None

def find_commands(inferior, tasks):
  first_offset, first_amount = u32(tasks,28),u32(tasks,24)
  for start,end in mappings():
    if end-start > 1<<24 or first_offset+first_amount*8 > end-start: continue
    try: commands = inferior.read_memory(start+first_offset,first_amount*8).tobytes()
    except gdb.MemoryError: continue
    known = sum(u64(commands,index*8)>>48 in TARGETS for index in range(first_amount))
    if known >= max(4,first_amount//2): return start
  return None

def dump_submit(address):
  inferior = gdb.selected_inferior()
  header = inferior.read_memory(address, SUBMIT_SIZE).tobytes()
  count, tasks_address = u32(header, 12), u64(header, 24)
  print(f"RKNPU_SUBMIT tasks={count} task_obj=0x{tasks_address:x}")
  task_va,tasks = find_tasks(inferior,count)
  if tasks is None:
    print("Unable to locate the mapped task descriptors")
    report = subprocess.run(["python3","dump.py","1","2"],cwd="/home/orangepi/npu/ops_rknn",text=True,
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
    print(report.stdout)
    return
  command_base = find_commands(inferior,tasks)
  print(f"mapped task_va=0x{task_va:x} command_va={('unknown' if command_base is None else hex(command_base))}")
  for task_index in range(count):
    task = tasks[task_index*TASK_SIZE:(task_index+1)*TASK_SIZE]
    op_index, enable, amount, command_address = u32(task,4),u32(task,8),u32(task,24),u64(task,32)
    print(f"TASK {task_index} op={op_index} enable=0x{enable:x} amount={amount} commands=0x{command_address:x}")
    if command_base is None: continue
    commands = inferior.read_memory(command_base+u32(task,28), amount*8).tobytes()
    for word_index in range(amount):
      word = u64(commands, word_index*8)
      target, value, register = word>>48, (word>>16)&0xffffffff, word&0xffff
      if target in TARGETS:
        marker = " BASE" if register in BASE_REGS else ""
        print(f"  {word_index:03d} {TARGETS[target]:8s} reg=0x{register:04x} value=0x{value:08x}{marker}")

class SubmitBreakpoint(gdb.Breakpoint):
  def stop(self):
    command, address = int(gdb.parse_and_eval("$x1")), int(gdb.parse_and_eval("$x2"))
    if ((command >> 8) & 0xff) == DRM_TYPE and (command & 0xff) == SUBMIT_NR:
      dump_submit(address)
    return False

SubmitBreakpoint("ioctl")
end

run
quit
