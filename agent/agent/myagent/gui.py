import tkinter as tk
from tkinter import ttk

class AgentConfigGUI:
    def __init__(self, env):
        self.env = env
        self.weights = {}
        self.text_input_value = ""
        self.tasks = self._get_tasks_from_env()
        
        self.root = tk.Tk()
        self.root.title("Agent Configuration")
        self.center_window(self.root, 400, 300)
        
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.current_frame = None
        self.show_main_menu()

    def _get_tasks_from_env(self):
        tasks = set()
        for recipe in self.env.recipes:
            if hasattr(recipe, 'contents_names'):
                 contents = recipe.contents_names
            else:
                 contents = [c.name for c in recipe.contents if hasattr(c, 'name')]
            
            ings_lower = [c.lower() for c in contents if c not in ['Plate']]
            if not ings_lower:
                continue
                
            for ing in ings_lower:
                tasks.add(f"chop_{ing}")
            
            possible_ings = ['lettuce', 'onion', 'tomato']
            recipe_ings = []
            for p in possible_ings:
                if p in ings_lower:
                    recipe_ings.append(p)
                    
            if not recipe_ings:
                recipe_ings = sorted(ings_lower)

            soup_name = '-'.join(recipe_ings) + ' soup'
            tasks.add(f"cook_{soup_name}")
            tasks.add(f"serve_{soup_name}")
            
        return sorted(list(tasks))

    def center_window(self, window, width, height):
        window.update_idletasks()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def clear_frame(self):
        if self.current_frame:
            self.current_frame.destroy()

    def show_main_menu(self):
        self.clear_frame()
        self.current_frame = ttk.Frame(self.main_frame)
        self.current_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(self.current_frame, text="Main Menu", font=("Arial", 16)).pack(pady=20)
        
        # Priority Weights Button
        ttk.Button(self.current_frame, text="Configure Priority Weights", 
                   command=self.show_weight_config).pack(pady=10, fill=tk.X)
        
        # Text Input
        ttk.Label(self.current_frame, text="Additional Settings (Text):").pack(pady=(20, 5), anchor="w")
        self.text_var = tk.StringVar(value=self.text_input_value)
        ttk.Entry(self.current_frame, textvariable=self.text_var).pack(fill=tk.X)
        
        # Start Game Button
        ttk.Button(self.current_frame, text="Start Game", command=self.finish).pack(pady=30, fill=tk.X)

    def show_weight_config(self):
        # Save text input before switching
        self.text_input_value = self.text_var.get()
        
        self.clear_frame()
        self.current_frame = ttk.Frame(self.main_frame)
        self.current_frame.pack(fill=tk.BOTH, expand=True)
        
        header = ttk.Frame(self.current_frame)
        header.pack(fill=tk.X)
        
        ttk.Button(header, text="< Back", command=self.show_main_menu).pack(side=tk.LEFT)
        ttk.Label(header, text="Priority Weights", font=("Arial", 14)).pack(side=tk.LEFT, padx=20)
        
        # Scrollable Area
        canvas = tk.Canvas(self.current_frame)
        scrollbar = ttk.Scrollbar(self.current_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        self.vars_dict = {}
        for t in self.tasks:
            row = ttk.Frame(scrollable_frame)
            row.pack(fill=tk.X, pady=2)
            
            lbl = ttk.Label(row, text=t, width=25)
            lbl.pack(side=tk.LEFT)
            
            # Use existing value or default 1.0
            val = self.weights.get(t, 1.0)
            var = tk.DoubleVar(value=val)
            self.vars_dict[t] = var
            
            scale = ttk.Scale(row, from_=0.1, to=10.0, variable=var, orient=tk.HORIZONTAL)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            entry = ttk.Entry(row, textvariable=var, width=5)
            entry.pack(side=tk.LEFT)

    def finish(self):
        # Save text input
        if hasattr(self, 'text_var'):
            self.text_input_value = self.text_var.get()
            
        # Save weights (if currently in weight screen, or generally from dict)
        if hasattr(self, 'vars_dict'):
             for t, var in self.vars_dict.items():
                self.weights[t] = var.get()
                
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return {
            'weights': self.weights,
            'text_input': self.text_input_value
        }

def configure_agent_settings(env):
    """
    GUIを起動して設定を取得する。
    戻り値: {'weights': dict, 'text_input': str}
    """
    gui = AgentConfigGUI(env)
    return gui.run()
