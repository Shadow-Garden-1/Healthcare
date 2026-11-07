
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib

# =====================================
# 1. LOAD MODEL AND DATASET
# =====================================

model = joblib.load("healthcare_model.pkl")
data = pd.read_csv("healthcare_dataset.csv")


# =====================================
# 2. COLORS
# =====================================

NAVY = "#14243A"
BLUE = "#2563EB"
BG = "#F1F5F9"
WHITE = "#FFFFFF"
TEXT = "#1E293B"
MUTED = "#64748B"
GREEN = "#059669"


# =====================================
# 3. MAIN WINDOW
# =====================================

root = tk.Tk()

root.title("Healthcare AI Assistant")
root.geometry("1150x750")
root.minsize(950, 650)
root.configure(bg=BG)


# =====================================
# 4. WIDGET STYLE
# =====================================

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "TCombobox",
    padding=8,
    fieldbackground=WHITE,
    background=WHITE
)

style.configure(
    "TEntry",
    padding=9
)


# =====================================
# 5. MAIN LAYOUT
# =====================================

sidebar = tk.Frame(root, bg=NAVY, width=220)
sidebar.pack(side="left", fill="y")

sidebar.pack_propagate(False)

main = tk.Frame(root, bg=BG)
main.pack(side="left", fill="both", expand=True)


# =====================================
# 6. SIDEBAR
# =====================================

tk.Label(
    sidebar,
    text="✚ HEALTH AI",
    font=("Arial", 20, "bold"),
    fg=WHITE,
    bg=NAVY
).pack(pady=(35, 10))

tk.Label(
    sidebar,
    text="INTELLIGENT HEALTHCARE",
    font=("Arial", 8, "bold"),
    fg="#94A3B8",
    bg=NAVY
).pack(pady=(0, 45))


def sidebar_item(text):

    tk.Label(
        sidebar,
        text=text,
        font=("Arial", 12),
        fg=WHITE,
        bg=NAVY,
        anchor="w",
        padx=25,
        pady=15
    ).pack(fill="x", pady=3)


sidebar_item("▦   Dashboard")
sidebar_item("♙   Patient Assessment")
sidebar_item("◈   ML Model")
sidebar_item("ⓘ   About")


tk.Label(
    sidebar,
    text="Powered by Python\nScikit-learn & Random Forest",
    font=("Arial", 9),
    fg="#94A3B8",
    bg=NAVY
).pack(side="bottom", pady=25)


# =====================================
# 7. HEADER
# =====================================

header = tk.Frame(main, bg=BG)
header.pack(fill="x", padx=35, pady=(25, 15))

tk.Label(
    header,
    text="Healthcare Dashboard",
    font=("Arial", 24, "bold"),
    fg=TEXT,
    bg=BG
).pack(anchor="w")

tk.Label(
    header,
    text="AI-powered experimental patient test result prediction",
    font=("Arial", 10),
    fg=MUTED,
    bg=BG
).pack(anchor="w", pady=5)


# =====================================
# 8. MODEL STATUS CARD
# =====================================

info = tk.Frame(main, bg=WHITE, padx=20, pady=15)
info.pack(fill="x", padx=35, pady=10)

tk.Label(
    info,
    text="MODEL STATUS",
    font=("Arial", 9, "bold"),
    fg=MUTED,
    bg=WHITE
).pack(anchor="w")

tk.Label(
    info,
    text="●  Random Forest Model Loaded",
    font=("Arial", 12, "bold"),
    fg=GREEN,
    bg=WHITE
).pack(anchor="w", pady=5)


# =====================================
# 9. CONTENT AREA
# =====================================

content = tk.Frame(main, bg=BG)
content.pack(fill="both", expand=True, padx=35, pady=15)


# LEFT FORM CARD

left = tk.Frame(
    content,
    bg=WHITE,
    padx=22,
    pady=20
)

left.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 12)
)


# RIGHT RESULT CARD

right = tk.Frame(
    content,
    bg=WHITE,
    padx=22,
    pady=20,
    width=280
)

right.pack(side="right", fill="y")
right.pack_propagate(False)


# =====================================
# 10. FORM HEADING
# =====================================

heading_frame = tk.Frame(left, bg=WHITE)
heading_frame.pack(fill="x")

tk.Label(
    heading_frame,
    text="Patient Information",
    font=("Arial", 17, "bold"),
    fg=TEXT,
    bg=WHITE
).pack(anchor="w")

tk.Label(
    heading_frame,
    text="Enter patient details below",
    font=("Arial", 9),
    fg=MUTED,
    bg=WHITE
).pack(anchor="w", pady=(5, 18))


# =====================================
# 11. INPUT FORM FRAME
# =====================================

form_frame = tk.Frame(left, bg=WHITE)
form_frame.pack(fill="x")

form_frame.grid_columnconfigure(0, weight=1)
form_frame.grid_columnconfigure(1, weight=1)

entries = {}


def create_field(parent, field, row, column):

    box = tk.Frame(parent, bg=WHITE)

    box.grid(
        row=row,
        column=column,
        sticky="ew",
        padx=8,
        pady=10
    )

    tk.Label(
        box,
        text=field,
        font=("Arial", 10, "bold"),
        fg=TEXT,
        bg=WHITE
    ).pack(anchor="w", pady=(0, 6))

    if field == "Age":

        widget = ttk.Entry(box)

    elif field == "Billing Amount":

        widget = ttk.Entry(box)

    else:

        values = sorted(
            data[field].dropna().astype(str).unique().tolist()
        )

        widget = ttk.Combobox(
            box,
            values=values,
            state="readonly"
        )

        if values:
            widget.current(0)

    widget.pack(fill="x", ipady=4)

    entries[field] = widget


# Create all input fields

create_field(form_frame, "Age", 0, 0)
create_field(form_frame, "Billing Amount", 0, 1)

create_field(form_frame, "Gender", 1, 0)
create_field(form_frame, "Blood Type", 1, 1)

create_field(form_frame, "Medical Condition", 2, 0)
create_field(form_frame, "Insurance Provider", 2, 1)

create_field(form_frame, "Admission Type", 3, 0)
create_field(form_frame, "Medication", 3, 1)


# =====================================
# 12. RESULT PANEL
# =====================================

tk.Label(
    right,
    text="Prediction Result",
    font=("Arial", 17, "bold"),
    fg=TEXT,
    bg=WHITE
).pack(anchor="w")

result_label = tk.Label(
    right,
    text="Waiting...",
    font=("Arial", 21, "bold"),
    fg=BLUE,
    bg=WHITE,
    wraplength=220
)

result_label.pack(anchor="w", pady=20)


probability_label = tk.Label(
    right,
    text="Enter patient details and click Predict.",
    font=("Arial", 10),
    fg=MUTED,
    bg=WHITE,
    justify="left",
    wraplength=220
)

probability_label.pack(anchor="w", pady=10)


# =====================================
# 13. PREDICTION FUNCTION
# =====================================

def predict():

    try:

        patient = pd.DataFrame([{

            "Age": int(entries["Age"].get()),

            "Billing Amount": float(
                entries["Billing Amount"].get()
            ),

            "Gender": entries["Gender"].get(),

            "Blood Type": entries["Blood Type"].get(),

            "Medical Condition": entries["Medical Condition"].get(),

            "Insurance Provider": entries["Insurance Provider"].get(),

            "Admission Type": entries["Admission Type"].get(),

            "Medication": entries["Medication"].get()

        }])

        result = model.predict(patient)[0]

        result_label.config(
            text=result,
            fg=BLUE
        )

        probabilities = model.predict_proba(patient)[0]

        probability_text = "MODEL PROBABILITY SCORES\n\n"

        for name, probability in zip(
            model.classes_, probabilities
        ):

            probability_text += (
                f"{name}: {probability * 100:.2f}%\n\n"
            )

        probability_label.config(
            text=probability_text
        )

    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter valid Age and Billing Amount."
        )

    except Exception as error:

        messagebox.showerror(
            "Prediction Error",
            str(error)
        )


# =====================================
# 14. CLEAR FUNCTION
# =====================================

def clear_fields():

    for field, widget in entries.items():

        if isinstance(widget, ttk.Combobox):

            if widget["values"]:
                widget.current(0)

        else:

            widget.delete(0, tk.END)

    result_label.config(
        text="Waiting...",
        fg=BLUE
    )

    probability_label.config(
        text="Enter patient details and click Predict."
    )


# =====================================
# 15. BUTTONS
# =====================================

button_frame = tk.Frame(left, bg=WHITE)

button_frame.pack(fill="x", pady=25)

tk.Button(
    button_frame,
    text="Predict Result",
    command=predict,
    bg=BLUE,
    fg=WHITE,
    activebackground="#1D4ED8",
    activeforeground=WHITE,
    font=("Arial", 11, "bold"),
    relief="flat",
    padx=22,
    pady=12,
    cursor="hand2"
).pack(side="left", padx=8)


tk.Button(
    button_frame,
    text="Clear",
    command=clear_fields,
    bg="#E2E8F0",
    fg=TEXT,
    activebackground="#CBD5E1",
    font=("Arial", 11, "bold"),
    relief="flat",
    padx=25,
    pady=12,
    cursor="hand2"
).pack(side="left", padx=8)


# =====================================
# 16. FOOTER
# =====================================

tk.Label(
    main,
    text="Educational project only | Not intended for medical diagnosis",
    font=("Arial", 9),
    fg=MUTED,
    bg=BG
).pack(pady=12)


# =====================================
# 17. RUN APPLICATION
# =====================================

root.mainloop()
