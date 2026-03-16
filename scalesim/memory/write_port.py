"""
External DRAM write requests serviced by Ramulator
"""
import numpy as np
import threading
from scalesim.scale_config import scale_config as config
from bisect import bisect_left

# This is shell module to ensure continuity

class write_port:
    """
    Class to define external DRAM memory requests with Ramulator
    """
    #
    def __init__(self):
        """
        __init__ module.
        """
        self.latency = 0
        self.ramulator_trace = False
        self.latency_matrix = []
        self.bw = 10
        self.request_queue_size = 100
        self.request_queue_status = 0
        self.stall_cycles = 0
        self.request_array = []
        self.count = 0
        self.config = config()
        self.num_banks = 16
        self.enable_bank_model = False
        self.enable_moe_parallel_bank_arb = False
        self.enable_dynamic_bank_alloc = False
        self.layer_name = ''
        self._service_lock = threading.Lock()
        self._reset_bank_model_state()

    def _reset_bank_model_state(self):
        """
        Reset per-bank bookkeeping state.
        """
        self.bank_busy_until = [0] * self.num_banks
        self.bank_busy_cycles = [0] * self.num_banks
        self.bank_access_count = [0] * self.num_banks
        self.bank_conflict_stall_cycles = 0
        self.bank_conflict_blocked_cycles = 0
        self.last_call_bank_conflict_blocked_cycles = 0
        self.queue_conflict_stall_cycles = 0
        self.total_requests_served = 0
        self.sim_start_cycle = None
        self.sim_stop_cycle = -1
    
    def def_params( self,
                    config = config(),
                    latency_file =''
                ):
        """
        Method to define the paths of ramulator trace numpy files 
        and write request queue sizes.
        """
        self.config = config
        self.ramulator_trace = self.config.get_ramulator_trace()
        self.request_queue_size = self.config.get_req_buf_sz_wr()
        self.bw = self.config.get_bandwidths_as_list()[0]
        self.enable_bank_model = self.config.get_enable_bank_model() if hasattr(self.config, 'get_enable_bank_model') else False
        self.enable_moe_parallel_bank_arb = self.config.get_enable_moe_parallel_bank_arb() if hasattr(self.config, 'get_enable_moe_parallel_bank_arb') else False
        self.enable_dynamic_bank_alloc = self.config.get_enable_dynamic_bank_alloc() if hasattr(self.config, 'get_enable_dynamic_bank_alloc') else False
        cfg_num_banks = self.config.get_num_bank() if hasattr(self.config, 'get_num_bank') else self.num_banks
        if cfg_num_banks is not None:
            self.num_banks = max(1, int(cfg_num_banks))
        if self.ramulator_trace == True:
            self.latency_matrix = np.load(latency_file)
        self.latency = 0
        self.request_array = []
        self.count = 0
        self._reset_bank_model_state()

    def set_bank_model_params(self,
                              num_banks=None,
                              enable_bank_model=None,
                              enable_moe_parallel_bank_arb=None,
                              enable_dynamic_bank_alloc=None,
                              request_queue_size=None,
                              layer_name='',
                              reset_state=True):
        """
        Optional lightweight control path for bank model parameters.
        """
        if num_banks is not None:
            self.num_banks = max(1, int(num_banks))

        if enable_bank_model is not None:
            self.enable_bank_model = bool(enable_bank_model)

        if enable_moe_parallel_bank_arb is not None:
            self.enable_moe_parallel_bank_arb = bool(enable_moe_parallel_bank_arb)

        if enable_dynamic_bank_alloc is not None:
            self.enable_dynamic_bank_alloc = bool(enable_dynamic_bank_alloc)

        if request_queue_size is not None:
            self.request_queue_size = max(1, int(request_queue_size))

        if layer_name is not None:
            self.layer_name = str(layer_name)

        if reset_state:
            self._reset_bank_model_state()
    #

    def find_latency(self):
        """
        Method to map DRAM return path latency for each transactions.
        """
        if(self.count < len(self.latency_matrix)):
            latency_out = self.latency_matrix[self.count]
            #print(str(self.count)+ ' ' + str(latency_out))
            self.count+=1
        else:
            latency_out = self.latency
        if(latency_out > 10000):
            latency_out = 0

        return latency_out

    def map_request_to_bank(self, req_addr):
        """
        Stable bank mapping using request address value from request matrix.
        """
        addr = int(req_addr)
        if addr < 0:
            addr = -addr
        return addr % self.num_banks

    def _prepare_inputs(self, incoming_requests_arr_np, incoming_cycles_arr):
        req_arr_np = np.asarray(incoming_requests_arr_np)
        cyc_arr_np = np.asarray(incoming_cycles_arr)

        if req_arr_np.ndim == 0:
            req_arr_np = req_arr_np.reshape((1, 1))
        elif req_arr_np.ndim == 1:
            req_arr_np = req_arr_np.reshape((-1, 1))

        cycles_flat = cyc_arr_np.reshape(-1)
        assert req_arr_np.shape[0] == cycles_flat.shape[0], 'Incoming cycles and requests dont match'
        return req_arr_np, cycles_flat, cyc_arr_np.shape

    def _is_parallel_arb_enabled(self):
        """
        MoE-only parallel arbitration gate.
        """
        if not self.enable_moe_parallel_bank_arb:
            return False
        return str(self.layer_name).startswith('MoE-')

    def _try_get_service_bank(self, req_addr, current_cycle, served_banks):
        """
        Select a bank for this request based on static/dynamic mode.
        """
        if self.enable_dynamic_bank_alloc:
            for bank_id in range(self.num_banks):
                bank_free = self.bank_busy_until[bank_id] <= current_cycle
                bank_not_used_in_cycle = bank_id not in served_banks
                if bank_free and bank_not_used_in_cycle:
                    return bank_id
            return None

        bank_id = self.map_request_to_bank(req_addr)
        bank_free = self.bank_busy_until[bank_id] <= current_cycle
        bank_not_used_in_cycle = bank_id not in served_banks
        if bank_free and bank_not_used_in_cycle:
            return bank_id
        return None

    def _service_without_bank_model(self, incoming_requests_arr_np, incoming_cycles_arr_np):
        """
        Compatibility path matching original queue-based timing behavior.
        """
        self.last_call_bank_conflict_blocked_cycles = 0
        _, cycles_flat, cycles_shape = self._prepare_inputs(
            incoming_requests_arr_np,
            incoming_cycles_arr_np
        )

        if self.ramulator_trace is False:
            out_cycles_arr = cycles_flat + self.latency
            if len(cycles_shape) == 2 and cycles_shape[1] == 1:
                return out_cycles_arr.reshape((-1, 1))
            return out_cycles_arr

        if len(cycles_flat) == 0:
            return cycles_flat

        updated_req_timestamp = int(cycles_flat[0])
        out_cycles_arr = np.zeros(cycles_flat.shape[0])
        for i in range(len(cycles_flat)):
            out_cycles_arr[i] = int(cycles_flat[i]) + self.stall_cycles + self.find_latency()
            self.request_array.append(out_cycles_arr[i])

            if len(self.request_array) == self.request_queue_size:
                updated_req_timestamp = int(cycles_flat[i]) + self.stall_cycles
                self.request_array.sort()
                if self.request_array[0] >= updated_req_timestamp:
                    self.stall_cycles += self.request_array[0] - updated_req_timestamp
                    updated_req_timestamp = self.request_array[0]
                    self.request_array.pop(0)
                else:
                    index = bisect_left(self.request_array, updated_req_timestamp)
                    if index == len(self.request_array):
                        self.request_array = []
                    else:
                        self.request_array = self.request_array[index:]
            elif len(self.request_array) > self.request_queue_size:
                self.request_array = self.request_array[-self.request_queue_size:]

        self.stall_cycles = 0
        if len(cycles_shape) == 2 and cycles_shape[1] == 1:
            return out_cycles_arr.reshape((-1, 1))
        return out_cycles_arr

    def _service_with_bank_model(self, incoming_requests_arr_np, incoming_cycles_arr):
        req_arr_np, cycles_flat, cycles_shape = self._prepare_inputs(
            incoming_requests_arr_np,
            incoming_cycles_arr
        )

        self.last_call_bank_conflict_blocked_cycles = 0

        out_cycles_arr = cycles_flat.astype(float).copy()
        pending_by_row = {}
        min_ready_cycle = None

        for row_id in range(req_arr_np.shape[0]):
            row_reqs = [int(x) for x in req_arr_np[row_id] if int(x) != -1]
            if len(row_reqs) == 0:
                continue
            pending_by_row[row_id] = row_reqs
            this_cycle = int(cycles_flat[row_id])
            if min_ready_cycle is None or this_cycle < min_ready_cycle:
                min_ready_cycle = this_cycle

        if len(pending_by_row) == 0:
            if len(cycles_shape) == 2 and cycles_shape[1] == 1:
                return out_cycles_arr.reshape((-1, 1))
            return out_cycles_arr

        queue_cap = max(int(self.request_queue_size), 1)
        active_rows = set(pending_by_row.keys())
        current_cycle = min_ready_cycle

        while len(active_rows) > 0:
            bank_conflict_this_cycle = False
            # Clear completed inflight requests to model queue occupancy.
            if len(self.request_array) > 0:
                self.request_array = [x for x in self.request_array if x > current_cycle]

            ready_rows_all = [rid for rid in sorted(active_rows) if int(cycles_flat[rid]) <= current_cycle]
            if self._is_parallel_arb_enabled():
                ready_rows = ready_rows_all
            elif len(ready_rows_all) > 0:
                # Non-MoE keeps serial request admission.
                ready_rows = [ready_rows_all[0]]
            else:
                ready_rows = []
            if len(ready_rows) == 0:
                current_cycle = min(int(cycles_flat[rid]) for rid in active_rows)
                continue

            served_banks = set()
            served_any = False

            for row_id in ready_rows:
                deferred_reqs = []
                for req_addr in pending_by_row[row_id]:
                    bank_id = self._try_get_service_bank(req_addr, current_cycle, served_banks)
                    bank_free = bank_id is not None
                    queue_free = len(self.request_array) < queue_cap

                    if bank_free and queue_free:
                        latency = int(self.find_latency()) if self.ramulator_trace else int(self.latency)
                        finish_cycle = current_cycle + latency
                        out_cycles_arr[row_id] = max(out_cycles_arr[row_id], finish_cycle)

                        self.request_array.append(finish_cycle)
                        self.bank_busy_until[bank_id] = current_cycle + 1
                        self.bank_busy_cycles[bank_id] += 1
                        self.bank_access_count[bank_id] += 1
                        self.total_requests_served += 1
                        served_banks.add(bank_id)
                        served_any = True
                    else:
                        deferred_reqs.append(req_addr)
                        if not queue_free:
                            self.queue_conflict_stall_cycles += 1
                        else:
                            self.bank_conflict_stall_cycles += 1
                            bank_conflict_this_cycle = True

                if len(deferred_reqs) == 0:
                    active_rows.remove(row_id)
                else:
                    pending_by_row[row_id] = deferred_reqs

            if bank_conflict_this_cycle:
                self.bank_conflict_blocked_cycles += 1
                self.last_call_bank_conflict_blocked_cycles += 1

            if not served_any:
                current_cycle += 1
                continue

            current_cycle += 1

        self.stall_cycles = self.bank_conflict_stall_cycles + self.queue_conflict_stall_cycles
        self.request_array = []

        if self.sim_start_cycle is None:
            self.sim_start_cycle = min_ready_cycle
        else:
            self.sim_start_cycle = min(self.sim_start_cycle, min_ready_cycle)

        self.sim_stop_cycle = max(self.sim_stop_cycle, int(np.max(out_cycles_arr)))

        if len(cycles_shape) == 2 and cycles_shape[1] == 1:
            return out_cycles_arr.reshape((-1, 1))

        return out_cycles_arr

    def service_writes(self, incoming_requests_arr_np, incoming_cycles_arr_np):
        """
        Method to service read request by the read buffer.
        Check for hit in the request queue or add the DRAM
        roundtrip latency for each transaction reported by 
        Ramulator.
        """
        with self._service_lock:
            if self.enable_bank_model:
                out_cycles_arr = self._service_with_bank_model(
                    incoming_requests_arr_np=incoming_requests_arr_np,
                    incoming_cycles_arr=incoming_cycles_arr_np
                )
                self.stall_cycles = 0
                return out_cycles_arr

            return self._service_without_bank_model(
                incoming_requests_arr_np=incoming_requests_arr_np,
                incoming_cycles_arr_np=incoming_cycles_arr_np
            )

    def get_total_sim_cycles(self):
        """
        Return local simulation cycle span tracked by this port.
        """
        if self.sim_start_cycle is None or self.sim_stop_cycle < self.sim_start_cycle:
            return 0
        return int(self.sim_stop_cycle - self.sim_start_cycle + 1)

    def get_bank_stats(self, total_sim_cycles=0):
        """
        Return per-bank busy/access/utilization records.
        """
        denom = int(total_sim_cycles)
        if denom <= 0:
            denom = self.get_total_sim_cycles()

        stats = []
        for bank_id in range(self.num_banks):
            busy = int(self.bank_busy_cycles[bank_id])
            acc = int(self.bank_access_count[bank_id])
            util = 0.0 if denom <= 0 else (busy / float(denom))
            stats.append({
                'bank_id': bank_id,
                'busy_cycles': busy,
                'access_count': acc,
                'utilization': util
            })
        return stats

    def get_bank_conflict_stall_cycles(self):
        """
        Return accumulated bank-conflict stall cycles (request-cycle units).
        """
        return int(self.bank_conflict_stall_cycles)

    def get_bank_conflict_blocked_cycles(self):
        """
        Return accumulated blocked cycles with at least one bank conflict in this port.
        """
        return int(self.bank_conflict_blocked_cycles)

    def get_last_call_bank_conflict_blocked_cycles(self):
        """
        Return blocked cycles caused by bank conflict in the most recent service call.
        """
        return int(self.last_call_bank_conflict_blocked_cycles)
