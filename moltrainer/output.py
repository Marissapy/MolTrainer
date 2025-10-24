"""
Output Formatter - GROMACS-style output with ASCII art logo and quotes
"""

import random
import shutil
from datetime import datetime
from art import text2art
from colorama import init, Fore, Style

# Initialize colorama for Windows support
init(autoreset=True)

from moltrainer.utils.quotes import QUOTES


class OutputFormatter:
    """Format output in GROMACS-style with logo, inputs, status, and quotes"""
    
    def __init__(self):
        # Auto-detect terminal width, with fallback
        try:
            terminal_width = shutil.get_terminal_size().columns
            # Use terminal width but limit between 60 and 120 for readability
            self.width = max(60, min(terminal_width, 120))
        except:
            self.width = 80  # Fallback to default
        
        self.separator = ":" + "-" * (self.width - 2) + ":"
    
    def print_header(self):
        """Print ASCII art header"""
        print("\n")
        print(self.separator)
        
        # Generate ASCII art logo (compact version)
        logo = text2art("MolTrainer", font="cybermedium")
        for line in logo.split('\n'):
            if line.strip():
                print(f": {line.center(self.width - 4)} :")
        
        print(f": {'':<{self.width - 4}} :")
        print(f": {Fore.CYAN}{'Machine Learning for Molecular Data'.center(self.width - 4)}{Style.RESET_ALL} :")
        print(f": {'Version 0.1.0'.center(self.width - 4)} :")
        print(self.separator)
        print()
    
    def print_inputs(self, inputs_dict):
        """Print user inputs in formatted style"""
        print(f"{Fore.CYAN}{'Input Parameters':^{self.width}}{Style.RESET_ALL}")
        print(self.separator)
        
        # Calculate dynamic column widths
        key_width = max(20, (self.width - 8) // 3)  # Min 20, ~1/3 of width
        value_width = self.width - key_width - 6
        
        for key, value in inputs_dict.items():
            print(f": {key:.<{key_width}} {str(value):.>{value_width}} :")
        
        print(self.separator)
        print()
        print(f"{Fore.YELLOW}Running Analysis...{Style.RESET_ALL}")
        print()
    
    def print_footer(self, success=True, error_msg=None, additional_info=None):
        """Print footer with status and random quote"""
        print()
        print(self.separator)
        
        # Calculate dynamic column widths
        key_width = max(20, (self.width - 8) // 3)
        value_width = self.width - key_width - 6
        
        # Print status
        if success:
            status = f"{Fore.GREEN}SUCCESS{Style.RESET_ALL}"
            print(f": {'Status':<{key_width}} {status:>{value_width}} :")
        else:
            status = f"{Fore.RED}FAILED{Style.RESET_ALL}"
            print(f": {'Status':<{key_width}} {status:>{value_width}} :")
            if error_msg:
                truncated_error = error_msg[:value_width]
                print(f": {'Error':<{key_width}} {truncated_error:>{value_width}} :")
        
        # Print additional info if provided
        if additional_info:
            print(f": {'':<{self.width - 4}} :")
            # Wrap long text
            for line in self._wrap_text(str(additional_info), self.width - 6):
                print(f": {line:<{self.width - 4}} :")
        
        print(self.separator)
        
        # Print timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f": {'Finished at':<{key_width}} {timestamp:>{value_width}} :")
        
        print(self.separator)
        
        # Print random quote
        self.print_random_quote()
        
        print(self.separator)
        print()
    
    def print_random_quote(self):
        """Print a random inspirational quote"""
        quote = random.choice(QUOTES)
        print(f": {'':<{self.width - 4}} :")
        
        # Wrap quote text
        for line in self._wrap_text(quote, self.width - 6):
            print(f": {line:<{self.width - 4}} :")
        
        print(f": {'':<{self.width - 4}} :")
    
    def print_error(self, message):
        """Print error message"""
        print(f"\n{Fore.RED}ERROR: {message}{Style.RESET_ALL}\n")
    
    def _wrap_text(self, text, max_width):
        """Wrap text to fit within max_width"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

