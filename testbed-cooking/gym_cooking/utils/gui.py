from tkinter import *

def popup_text(description: str = '', fill_in: str = '') -> str | None:
    root = Tk()
    # root.geometry("300x300")
    root.title("Input Window")

    ret = None

    def take_input():
        i = inputtxt.get("1.0", "end-1c")
        nonlocal ret
        ret = i
        root.destroy()

    label = Label(text=description, font=('Times New Roman', 15, 'bold'))
    inputtxt = Text(root, height=15, width=30, bg="light yellow", font=('Times New Roman', 15, 'bold'))
    inputtxt.insert("end-1c", fill_in)
    display = Button(root, height=2, width=5, text="Submit", command=lambda: take_input())

    label.pack()
    inputtxt.pack()
    display.pack()

    mainloop()

    return ret


def popup_choice(description: str, choices: list[str]) -> str | None:
    root = Tk()
    root.geometry("300x200")
    root.title("Choose Window")

    ret = None

    def take_input():
        nonlocal ret
        ret = var.get()
        root.destroy()

    label = Label(text=description, font=('Times New Roman', 15, 'bold'))
    var = StringVar(root)
    var.set(choices[0])
    option = OptionMenu(root, var, *choices)
    display = Button(root, height=2, width=5, text="Submit", command=lambda: take_input())

    label.pack()
    option.pack(expand=True)
    display.pack()

    mainloop()

    return ret


def popup_task_choice(description: str, choices: list) -> str | None:
    """
    Show a task selection UI.

    Choices may be either plain strings (legacy) or tuples/lists of (display_str, payload).
    Returns the selected item as follows:
      - If input items are plain strings, returns the selected string (unchanged).
      - If input items are (display, payload), returns the tuple (display, payload).
    """
    if not choices:
        return None

    root = Tk()
    root.geometry("480x420")
    root.title("Select Task")

    ret = None

    def submit_selection():
        nonlocal ret
        selected = listbox.curselection()
        if not selected:
            return
        choice = choices[selected[0]]
        # Return the original structure: string or tuple
        ret = choice
        root.destroy()

    label = Label(text=description, font=('Times New Roman', 13, 'bold'))
    listbox = Listbox(root, width=60, height=16, font=('Times New Roman', 12))
    scrollbar = Scrollbar(root, orient=VERTICAL, command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)

    for item in choices:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            display = item[0]
        else:
            display = str(item)
        listbox.insert(END, display)

    if choices:
        listbox.selection_set(0)

    listbox.bind('<Double-Button-1>', lambda _e: submit_selection())

    btn_frame = Frame(root)
    submit_btn = Button(btn_frame, height=2, width=10, text="Submit", command=submit_selection)
    cancel_btn = Button(btn_frame, height=2, width=10, text="Cancel", command=root.destroy)

    label.pack(pady=8)
    listbox.pack(side=LEFT, fill=BOTH, expand=True, padx=(12, 0), pady=8)
    scrollbar.pack(side=LEFT, fill=Y, pady=8, padx=(0, 12))
    btn_frame.pack(pady=(0, 10))
    submit_btn.pack(side=LEFT, padx=6)
    cancel_btn.pack(side=LEFT, padx=6)

    mainloop()

    return ret

def popup_box(description: str = '') -> bool | None:
    root = Tk()
    # root.geometry("300x300")
    root.title("Box Window")

    ret = None

    def take_yes():
        nonlocal ret
        ret = True
        root.destroy()

    def take_no():
        nonlocal ret
        ret = False
        root.destroy()

    label = Label(text=description, font=('Times New Roman', 15, 'bold'))
    display_yes = Button(root, height=2, width=5, text="Yes", command=lambda: take_yes())
    display_no = Button(root, height=2, width=5, text="No", command=lambda: take_no())

    label.pack()
    display_yes.pack()
    display_no.pack()

    mainloop()

    return ret

if __name__ == '__main__':
    print(popup_text("test", "test"))
    print(popup_choice("tes11111t", ["test1", "test2"]))
    print(popup_box("tes22222"))

