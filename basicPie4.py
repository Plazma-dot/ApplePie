#   BasicPIE4 Made by Plazma-dot
#   
#   Original Github Project: https://github.com/Plazma-dot/BasicPie
#   
#   This project is under MIT license
#
#                               12/30/2024
#--------------------------------------------------------

import os
import re
import sys
from typing import List, Tuple, Union

# Tokenizer and Math handling code from the second snippet
DIGITS = '0123456789'

# Tokens
TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_KEYWORD = 'KEYWORD'
KEYWORDS = ['PRINT', 'INPUT', 'LET', 'RUN', 'LOAD', 'SAVE', 'LIST', 'GOTO', 'GOSUB', 'RETURN', 'END', 'IF', 'THEN']
TT_IDENTIFIER = 'IDENTIFIER'
TT_STRING = 'STRING'

class Token:
    def __init__(self, type_, value=None):
        self.type = type_
        self.value = value
    
    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'

class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()
    
    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        tokens = []

        while self.current_char is not None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char.isalpha():  # Handle alphabetic characters
                tokens.append(self.make_identifier())
            elif self.current_char == '"':  # Handle string literals
                tokens.append(self.make_string())
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN))
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        return tokens, None

    def make_string(self):
        string_value = ''
        self.advance()  # Skip the opening quote
        while self.current_char is not None and self.current_char != '"':
            string_value += self.current_char
            self.advance()
        self.advance()  # Skip the closing quote
        return Token(TT_STRING, string_value)


    def make_number(self):
        num_str = ''
        dot_count = 0

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str))
        else:
            return Token(TT_FLOAT, float(num_str))
    
    def make_identifier(self):
        identifier_str = ''

        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            identifier_str += self.current_char
            self.advance()

        token_type = TT_KEYWORD if identifier_str.upper() in KEYWORDS else TT_IDENTIFIER
        return Token(token_type, identifier_str.upper())


class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details
    
    def as_string(self):
        result  = f'{self.error_name}: {self.details}\n'
        result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
        return result

class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)


# Main logic code
# Clear the console
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_console()

print(".______     ___      __________  ______.   .__________ ._______   _  _   ")
print("|   _  \   /   \    /   ___/   |/  ____|   |   _      ||   ____| | || |_.")
print("|   _  < /  /_\  \  \   \   |     |        |   ___/|  ||   __|   |__   _|")
print("|  |_)  |  _____  ---)   |  |     ----.    |  |    |  ||  |____     | |  ")
print("|________/     \_______/   |__|\______|    | _|    |___________|    |_|  ")
print("")

DEFAULT_SAVE_DIR = os.getcwd()
os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)

MEM_SLOT: List[Tuple[int, str, List[str]]] = []

# Tokenizer (Lexer)
def tokenize(line: str):
    lexer = Lexer('<stdin>', line)
    tokens, error = lexer.make_tokens()
    if error:
        print(error.as_string())
        return []
    return tokens

def parse_line(tokens: List[Token]) -> Tuple[Union[int, None], str, List[str]]:
    if not tokens or tokens[0] is None:
        raise ValueError("Invalid tokens provided to parse_line.")
    line_number = int(tokens.pop(0).value) if tokens[0].type == TT_INT else None
    if not tokens or tokens[0].type != TT_KEYWORD:
        raise ValueError("Command token is missing or invalid.")
    command = tokens.pop(0).value.upper()
    args = [token.value for token in tokens if token is not None]
    return line_number, command, args


def evaluate_expression(expression: str, variables: dict):
    lexer = Lexer('<stdin>', expression)
    tokens, error = lexer.make_tokens()
    if error:
        print(error.as_string())
        return None
    # Here you could add parsing logic to handle mathematical operations manually
    # For simplicity, we can use eval, but it's not ideal for production.
    return eval(expression, {}, variables)

def execute(command: str, args: List[str], variables: dict, line_index: int, 
            lines: List[Tuple[int, str, List[str]]], call_stack: List[int]) -> Tuple[bool, int]:
    command = command.upper()

    if command == "PRINT":
        output = " ".join(
            arg.strip('"') if arg.startswith('"') and arg.endswith('"') else str(variables.get(arg, arg))
            for arg in args
        )
        print(output)

    elif command == "INPUT":
        prompt = " ".join(args[:-1]).replace('"', '')
        var_name = args[-1]
        variables[var_name] = input(f"{prompt} ")

    elif command == "LET":
        var_name = args[0]
        expression = " ".join(args[2:])
        try:
            variables[var_name] = evaluate_expression(expression, variables)
        except Exception as e:
            print(f"Error evaluating LET expression: {e}")

    elif command == "HOME":
        clear_console()
        return False, line_index

    elif command == "RUN":
        run_program()

    elif command == "LOAD":
        load_program(args[0])

    elif command == "SAVE":
        save_program(args[0], args[1])

    elif command == "LIST":
        if lines:
            for line in sorted(lines, key=lambda x: x[0]):
                    return f"{line[0]} {' '.join([line[1]] + line[2])}"
        else:
            return ""

    elif command == "GOTO":
        target_line = int(args[0])
        target_index = next((i for i, line in enumerate(lines) if line[0] == target_line), None)
        return (True, target_index) if target_index is not None else (print(f"Line {target_line} not found."), line_index)

    elif command == "GOSUB":
        target_line = int(args[0])
        target_index = next((i for i, line in enumerate(lines) if line[0] == target_line), None)
        if target_index is not None:
            call_stack.append(line_index + 1)
            return True, target_index
        print(f"Line {target_line} not found.")
        return False, line_index

    elif command == "RETURN":
        if call_stack:
            return True, call_stack.pop()
        print("RETURN called without matching GOSUB.")

    elif command == "END":
        return False, line_index

    elif command == "IF":
        try:
            condition, then_command = " ".join(args).split(" THEN ", 1)
            
            condition_result = evaluate_expression(condition, variables)
            if condition_result:
                new_tokens = tokenize(then_command)
                _, then_command, then_args = parse_line(new_tokens)
                return execute(then_command, then_args, variables, line_index, lines, call_stack)
        except Exception as e:
            print(f"Error evaluating IF condition: {e}")

    else:
        print(f"Unknown command: {command}")
    
    return True, line_index

def run_program(lines: List[Tuple[int, str, List[str]]]):
    variables = {}
    call_stack = []
    line_index = 0

    while line_index < len(lines):
        line_number, command, args = lines[line_index]
        continue_execution, new_index = execute(command, args, variables, line_index, lines, call_stack)
        if not continue_execution:
            break
        line_index = new_index if new_index != line_index else line_index + 1

def save_program(lines: List[Tuple[int, str, List[str]]], directory: str, filename: str):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        for line in lines:
            f.write(f"{line[0]} {' '.join([line[1]] + line[2])}\n")
    print(f"Program saved to {filepath}")

def load_program(directory: str, filename: str) -> List[Tuple[int, str, List[str]]]:
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        return []

    with open(filepath, "r") as f:
        lines = [parse_line(tokenize(line.strip())) for line in f]
    print(f"Program loaded from {filepath}")
    return lines

def main():
    lines = []

    while True:
        user_input = input("BASIC> ").strip()

        if user_input.upper() == "RUN":
            if lines:
                run_program(lines)
            else:
                print("Error: No program to run.")

        elif user_input.upper() == "LIST":
            if lines:
                for line in sorted(lines, key=lambda x: x[0]):
                    print(f"{line[0]} {' '.join([line[1]] + line[2])}")
            else:
                print("No lines to list.")

        elif user_input.upper() == "HOME":
            clear_console()

        elif user_input.upper() == "EXIT":
            break

        elif user_input.upper().startswith("SAVE "):
            path_filename = user_input[5:].strip('"')
            directory, filename = os.path.split(path_filename) or (DEFAULT_SAVE_DIR, path_filename)
            save_program(lines, directory or DEFAULT_SAVE_DIR, filename)

        elif user_input.upper().startswith("LOAD "):
            path_filename = user_input[5:].strip('"')
            directory, filename = os.path.split(path_filename) or (DEFAULT_SAVE_DIR, path_filename)
            lines = load_program(directory or DEFAULT_SAVE_DIR, filename)

        else:
            tokens = tokenize(user_input)
            if tokens:
                parsed_line = parse_line(tokens)
                if parsed_line[0] is not None:
                    lines = [line for line in lines if line[0] != parsed_line[0]] + [parsed_line]
                    lines.sort(key=lambda x: x[0])
                else:
                    print("Error: Invalid syntax.")

if __name__ == "__main__":
    main()
