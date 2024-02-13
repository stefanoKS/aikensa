from PyQt5.QtCore import QThread, pyqtSignal
import socket
import time

class ServerMonitorThread(QThread):
    server_status_signal = pyqtSignal(bool)  # True if server is up, False if down
    input_states_signal = pyqtSignal(list)  # Signal to emit the input states

    def __init__(self, server_ip, server_port, check_interval=1):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.check_interval = check_interval
        self.running = True
        self.sock = None

    def reconnect(self):
        self.sock.close()  # Close the existing socket
        time_to_wait = 1  # Start with a 1-second wait
        max_wait = 32  # Maximum wait time of 32 seconds
        while self.running:
            try:
                print(f"Attempting to reconnect in {time_to_wait} seconds...")
                time.sleep(time_to_wait)
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.server_ip, self.server_port))
                self.sock.settimeout(3)
                print("Reconnected successfully.")
                break  # Exit the loop on successful reconnection
            except socket.error:
                print("Reconnection failed. Trying again...")
                time_to_wait = min(time_to_wait * 2, max_wait)


    def binary_states_to_hex_command(self, input_states):
        hex_digits = []
        for i in range(0, len(input_states), 4):

            binary_slice = input_states[i:i+4]
            binary_slice.reverse()

            hex_digit = hex(int(''.join(str(bit) for bit in binary_slice), 2))[2:].upper()
            hex_digits.append(hex_digit)

        hex_string = ''.join(hex_digits)
        command = f"@W04{hex_string}000000000000\r\n"
        
        return command

    def hex_to_binary(self, hex_value):
        return bin(int(hex_value, 16))[2:].zfill(4)

    def parse_io_states(self, response):
        hex_data = response[4:-2]  

        binary_data = ''.join(self.hex_to_binary(hex_digit) for hex_digit in hex_data)

        input_states = ''.join(binary_data[i:i+4][::-1] for i in range(0, 16, 4))
        output_states = ''.join(binary_data[i:i+4][::-1] for i in range(16, 32, 4))

        input_states = [int(bit) for bit in input_states]
        output_states = [int(bit) for bit in output_states]

        # print("Input states:", input_states)
        # print("Output states:", output_states)
        
        return input_states, output_states
        
    def send_command(self, command):
        try:
            self.sock.send(command.encode('utf-8'))
            response = b''
            while True:
                part = self.sock.recv(1024)
                response += part
                if b'\r\n' in response:
                    break
            return response.decode('utf-8')
        except socket.timeout:
            print("Socket timeout occurred. Attempting to reconnect...")
            self.server_status_signal.emit(False)  # Emit False upon timeout
            self.reconnect()
            return ""  # Return an empty string or handle as needed
        except OSError as e:
            print(f"An OSError occurred: {e}. Attempting to reconnect...")
            self.server_status_signal.emit(False)  # Emit False upon other socket errors
            self.reconnect()
            return ""  # Return an empty string or handle as needed


    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.server_ip, self.server_port))
            self.sock.settimeout(3)
            # self.server_status_signal.emit(True)

            while self.running:
                # Send read command and process response
                read_command = '@R01\r\n'
                response = self.send_command(read_command)
                # print("Received I/O response:", response)
                
                #if there is response, emit signal
                if response:
                    self.server_status_signal.emit(True)

                input_states, output_states = self.parse_io_states(response)
                self.input_states_signal.emit(input_states)
                
                # Example to write based on input states (customize as needed)
                # Here you could decide when to write based on input states or other logic

                EtherCommand = self.binary_states_to_hex_command(input_states)
                # print ("send ether Command:", EtherCommand)
                ether_response = self.send_command(EtherCommand)
                # print ("Received Ether Response:", ether_response)

                time.sleep(self.check_interval)

        except Exception as e:
            print("An error occurred:", e)
            self.server_status_signal.emit(False)
        finally:
            if self.sock:
                self.sock.close()

    def stop(self):
        self.running = False
