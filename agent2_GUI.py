from dotenv import load_dotenv
import os
import asyncio
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import threading
from agents import Agent, Runner, WebSearchTool
from datetime import datetime
import json

load_dotenv()

def load_prompt(file_name):
    base_dir = "/Users/zac/Desktop/master_thesis/code/1_instructions"
    file_path = os.path.join(base_dir, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

class Agent2GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Creative Idea Generator - Unlimited Text Input")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # consturct Agent 2 
        self.agent = Agent(
            name="Creative Idea Generator",
            instructions=load_prompt("agent2_ver2_1.txt")
        )
        
        self.history = []
        self.is_processing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # title
        title_label = tk.Label(
            main_frame, 
            text="🎨 Creative Idea Generator", 
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=(0, 10))
        
        # conversation display
        chat_frame = tk.Frame(main_frame, bg='#ffffff', relief=tk.SUNKEN, bd=2)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=80,
            height=25,
            font=("Arial", 11),
            bg='#ffffff',
            fg='#333333',
            state=tk.DISABLED,
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure text tags for styling
        self.chat_display.tag_configure("user", foreground="#0066cc", font=("Arial", 11, "bold"))
        self.chat_display.tag_configure("assistant", foreground="#009900", font=("Arial", 11))
        self.chat_display.tag_configure("system", foreground="#666666", font=("Arial", 10, "italic"))
        self.chat_display.tag_configure("error", foreground="#cc0000", font=("Arial", 11))
        
        # Input area
        input_frame = tk.Frame(main_frame, bg='#f0f0f0')
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input label and character count
        input_label_frame = tk.Frame(input_frame, bg='#f0f0f0')
        input_label_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(
            input_label_frame, 
            text="Describe your idea or problem (supports unlimited text length):", 
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        ).pack(side=tk.LEFT)
        
        self.char_count_label = tk.Label(
            input_label_frame, 
            text="Character count: 0", 
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#666666'
        )
        self.char_count_label.pack(side=tk.RIGHT)
        
        # Text input box
        self.text_input = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            width=80,
            height=8,
            font=("Arial", 11),
            bg='#ffffff',
            fg='#333333',
            relief=tk.SUNKEN,
            bd=2,
            padx=10,
            pady=10
        )
        self.text_input.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        self.text_input.bind('<KeyRelease>', self.update_char_count)
        self.text_input.bind('<Control-Return>', self.send_message)
        
        # Button area
        button_frame = tk.Frame(input_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X)
        
        # Send button
        self.send_button = tk.Button(
            button_frame,
            text="Generate Ideas (Ctrl+Enter)",
            command=self.send_message,
            bg='#007bff',
            fg='white',
            font=("Arial", 12, "bold"),
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=5
        )
        self.send_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Function buttons
        tk.Button(
            button_frame,
            text="Clear Chat",
            command=self.clear_chat,
            bg='#dc3545',
            fg='white',
            font=("Arial", 10),
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=5
        ).pack(side=tk.LEFT)
        
        tk.Button(
            button_frame,
            text="Save Chat",
            command=self.save_chat,
            bg='#28a745',
            fg='white',
            font=("Arial", 10),
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        tk.Button(
            button_frame,
            text="Load File",
            command=self.load_file,
            bg='#ffc107',
            fg='black',
            font=("Arial", 10),
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        # Status bar
        self.status_label = tk.Label(
            main_frame,
            text="Ready",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#666666',
            anchor='w'
        )
        self.status_label.pack(fill=tk.X, pady=(5, 0))
        
        # Initialize message - 修改为Agent 2相关
        self.add_message("System", "🎨 Welcome to Creative Idea Generator! Describe your assistive technology idea or problem, and I'll suggest creative alternatives and improvements.", "system")
        
    def update_char_count(self, event=None):
        """Update character count"""
        text = self.text_input.get("1.0", tk.END).strip()
        char_count = len(text)
        self.char_count_label.config(text=f"Character count: {char_count}")
        
    def add_message(self, sender, message, tag=""):
        """Add message to chat display area"""
        self.chat_display.config(state=tk.NORMAL)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if sender == "User":
            self.chat_display.insert(tk.END, f"[{timestamp}] 👤 You: ", "user")
        elif sender == "Assistant":
            self.chat_display.insert(tk.END, f"[{timestamp}] 🎨 Creative Assistant: ", "assistant")
        else:
            self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: ", tag)
        
        self.chat_display.insert(tk.END, f"{message}\n\n", tag)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
    def send_message(self, event=None):
        """Send message"""
        if self.is_processing:
            return
            
        message = self.text_input.get("1.0", tk.END).strip()
        if not message:
            messagebox.showwarning("Warning", "Please enter a message")
            return
            
        # Display user message
        self.add_message("User", message, "user")
        
        # Clear input box
        self.text_input.delete("1.0", tk.END)
        self.update_char_count()
        
        # Disable send button
        self.is_processing = True
        self.send_button.config(state=tk.DISABLED, text="Generating Ideas...")
        self.status_label.config(text="Generating creative ideas...")
        
        # Process in background thread
        thread = threading.Thread(target=self.process_message, args=(message,))
        thread.daemon = True
        thread.start()
        
    def process_message(self, message):
        """Process message in background"""
        try:
            # Build prompt
            prompt = ""
            for role, msg in self.history:
                prompt += f"{role}: {msg}\n"
            prompt += f"User: {message}\nAssistant:"
            
            # Run model
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(Runner.run(self.agent, prompt))
            reply = result.final_output.strip()
            
            # Update history
            self.history.append(("User", message))
            self.history.append(("Assistant", reply))
            
            # Update UI in main thread
            self.root.after(0, self.update_ui_after_processing, reply, False)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            self.root.after(0, self.update_ui_after_processing, error_msg, True)
            
    def update_ui_after_processing(self, message, is_error=False):
        """Update UI after processing is complete"""
        if is_error:
            self.add_message("Error", message, "error")
        else:
            self.add_message("Assistant", message, "assistant")
            
        # Restore send button
        self.is_processing = False
        self.send_button.config(state=tk.NORMAL, text="Generate Ideas (Ctrl+Enter)")
        self.status_label.config(text="Ready")
        
    def clear_chat(self):
        """Clear chat"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all chat history?"):
            self.history.clear()
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.add_message("System", "Chat cleared", "system")
            
    def save_chat(self):
        """Save chat to file"""
        if not self.history:
            messagebox.showinfo("Info", "No chat history to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    if file_path.endswith('.json'):
                        json.dump(self.history, f, ensure_ascii=False, indent=2)
                    else:
                        f.write("=== Creative Idea Generator Chat History ===\n\n")
                        for i, (role, msg) in enumerate(self.history, 1):
                            f.write(f"[{i}] {role}:\n{msg}\n\n")
                            
                messagebox.showinfo("Success", f"Chat history saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving file: {str(e)}")
                
    def load_file(self):
        """Load content from file to input box"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                self.text_input.delete("1.0", tk.END)
                self.text_input.insert("1.0", content)
                self.update_char_count()
                
                messagebox.showinfo("Success", f"File content loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error reading file: {str(e)}")

def main():
    root = tk.Tk()
    app = Agent2GUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Program exited")

if __name__ == "__main__":
    main()