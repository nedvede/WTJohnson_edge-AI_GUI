import subprocess

def execute_ampy_command(command):
  """Executes an ampy command and returns the output."""
  process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
  output, _ = process.communicate()
  return output.decode("utf-8")

def main():
  """Executes the ampy command and prints the output."""
  output = execute_ampy_command("ampy --port COM3 run C:/Line_detection_leeds/pico/test.py")
  print(output)

if __name__ == "__main__":
  main()
