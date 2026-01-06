import tkinter as tk
from tkinter import ttk

def configure_task_weights(env):
    """
    環境から可能なタスク一覧を取得し、GUIで重みを設定させる。
    戻り値: {task_key: weight (float)}
    """
    
    # 1. 環境からタスクリストを生成
    tasks = set()
    
    # env.recipes には Recipe クラスのインスタンスが入っている
    for recipe in env.recipes:
        # レシピに含まれる食材名を取得 (e.g. ['Tomato', 'Onion'])
        # recipe.contents は Food オブジェクトのリスト
        # Food.name は 'Tomato' 等
        if hasattr(recipe, 'contents_names'):
             # すでに文字列リストがある場合
             contents = recipe.contents_names
        else:
             contents = [c.name for c in recipe.contents if hasattr(c, 'name')]
             
        # Lowercase ingredients
        ings_lower = [c.lower() for c in contents if c not in ['Plate']]
        if not ings_lower:
            continue
            
        # Chop tasks
        for ing in ings_lower:
            tasks.add(f"chop_{ing}")
            
        # Soup name construction (consistent with CSPAgent)
        # ings_lower はレシピ定義順だが、CSPAgentではアルファベット順などで結合している可能性がある
        # CSPAgent: ings_lower = [ing for ing in ['lettuce','onion','tomato'] if ing in name]
        # つまり 'lettuce', 'onion', 'tomato' の固定順序でフィルタリングしている。
        
        # ここでも合わせる
        possible_ings = ['lettuce', 'onion', 'tomato']
        recipe_ings = []
        for p in possible_ings:
            if p in ings_lower:
                # 数が含まれている場合に対応（例：トマト2個）
                # しかしCSPAgentは単純な存在チェックしかしていない: if ing in name
                # なので1回だけ追加される
                recipe_ings.append(p)
                
        if not recipe_ings:
            # Maybe single ingredient soup or other type?
            # Fallback to sorted list
            recipe_ings = sorted(ings_lower)

        soup_name = '-'.join(recipe_ings) + ' soup'
        
        tasks.add(f"cook_{soup_name}")
        tasks.add(f"serve_{soup_name}")

    sorted_tasks = sorted(list(tasks))
    
    # 2. GUI構築
    root = tk.Tk()
    root.title("Task Weight Configuration")
    
    weights = {}
    
    frame = ttk.Frame(root, padding="10")
    frame.pack(fill=tk.BOTH, expand=True)
    
    ttk.Label(frame, text="Adjust Priority Weights for Tasks").pack(pady=5)
    
    canvas = tk.Canvas(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    vars_dict = {}

    def create_row(task_name):
        row = ttk.Frame(scrollable_frame)
        row.pack(fill=tk.X, pady=2)
        
        lbl = ttk.Label(row, text=task_name, width=30)
        lbl.pack(side=tk.LEFT)
        
        var = tk.DoubleVar(value=1.0)
        vars_dict[task_name] = var
        
        # Slider
        scale = ttk.Scale(row, from_=0.1, to=10.0, variable=var, orient=tk.HORIZONTAL)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Entry
        entry = ttk.Entry(row, textvariable=var, width=5)
        entry.pack(side=tk.LEFT)
        
    for t in sorted_tasks:
        create_row(t)
        
    def on_start():
        for t, var in vars_dict.items():
            weights[t] = var.get()
        root.destroy()
        
    btn = ttk.Button(root, text="Start Game", command=on_start)
    btn.pack(pady=10)
    
    # Center window
    root.update_idletasks()
    width = 500
    height = 600
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    root.mainloop()
    
    return weights
