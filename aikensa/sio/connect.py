import socket
import time

# SiO controller IP address and port number from the settings image
HOST = '192.168.0.100'  # Use the IP address from SiO settings
PORT = 30001            # Use the port number from SiO settings

command = '@R01' + '\r\n'  # '@R01' followed by CR and LF

def binary_states_to_hex_command(input_states):
    hex_digits = []
    for i in range(0, len(input_states), 4):

        binary_slice = input_states[i:i+4]
        binary_slice.reverse()

        hex_digit = hex(int(''.join(str(bit) for bit in binary_slice), 2))[2:].upper()
        hex_digits.append(hex_digit)

    hex_string = ''.join(hex_digits)
    command = f"@W04{hex_string}000000000000\r\n"
    
    return command

def hex_to_binary(hex_value):
    return bin(int(hex_value, 16))[2:].zfill(4)

def parse_io_states(response):
    hex_data = response[4:-2]  

    binary_data = ''.join(hex_to_binary(hex_digit) for hex_digit in hex_data)

    input_states = ''.join(binary_data[i:i+4][::-1] for i in range(0, 16, 4))
    output_states = ''.join(binary_data[i:i+4][::-1] for i in range(16, 32, 4))

    input_states = [int(bit) for bit in input_states]
    output_states = [int(bit) for bit in output_states]

    print("Input states:", input_states)
    print("Output states:", output_states)
    
    return input_states, output_states

# def send_command(sock, command):
#     sock.send(command.encode('utf-8'))
#     response = sock.recv(1024).decode('utf-8')
#     print ("Current response:", response)
#     return response

def send_command(sock, command):
    sock.send(command.encode('utf-8'))
    response = b''
    
    while True:
        part = sock.recv(1024)
        response += part
        print("Current response:", response)
        if b'\r\n' in response:  # Check the accumulated response
            break
    
    return response.decode('utf-8')

if __name__ == '__main__':
    # Initialize socket outside of the send_command function
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    sock.settimeout(3)

    try:
        # Example usage
        while True:
            print("send command:", command)
            response = send_command(sock, command)
            print("Received I/O response:", response)
            input_states, output_states = parse_io_states(response)
            

            time.sleep(0.05) 

            EtherCommand = binary_states_to_hex_command(input_states)
            print ("send ether Command:", EtherCommand)
            ether_response = send_command(sock, EtherCommand)
            print ("Received Ether Response:", ether_response)
            # sendOutput = send_command(sock, OutputCommand)
            # print ("Received Output Response:", sendOutput)

            time.sleep(0.05)


    except Exception as e:
        print("An error occurred:", e)
    finally:
        sock.close()  # Ensure the socket is closed when done
