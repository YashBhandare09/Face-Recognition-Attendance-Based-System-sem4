import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import csv
import numpy as np
import datetime
import pickle

# Directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
training_images_dir = os.path.join(current_dir, "training_images")
training_images_and_labels_dir = os.path.join(current_dir, "training_images_and_labels")
student_details_file = os.path.join(current_dir, "student_details.csv")
teacher_details_file = os.path.join(current_dir, "teacher_details.csv")
student_attendance_file = os.path.join(current_dir, "student_attendance.csv")
teacher_attendance_file = os.path.join(current_dir, "teacher_attendance.csv")
trainer_file = os.path.join(current_dir, "trainer.yml")
labels_file = os.path.join(current_dir, "labels.pickle")

# Ensure directories exist
os.makedirs(training_images_dir, exist_ok=True)
os.makedirs(training_images_and_labels_dir, exist_ok=True)

# Initialize Tkinter
root = tk.Tk()
root.title("Attendance System")
root.geometry("1200x800")  # Adjusted window size for better viewing

# Function to capture images for training
def capture_images_for_training(name, num_images=50):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access the webcam.")
        return []

    count = 0
    images = []

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            img_name = f"{name}_{count}.jpg"
            img_path = os.path.join(training_images_dir, img_name)
            cv2.imwrite(img_path, roi_gray)
            images.append(img_path)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Capturing Images', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return images

# Function to label captured images and save in appropriate directory
def label_images(images, label):
    label_dir = os.path.join(training_images_and_labels_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    for img_path in images:
        img_name = os.path.basename(img_path)
        new_img_path = os.path.join(label_dir, img_name)
        os.rename(img_path, new_img_path)

# Function to save student profile
def save_student_profile():
    name = student_name.get()
    course = student_course.get()
    if not name or not course:
        messagebox.showerror("Error", "Please enter both Name and Course")
        return

    images = capture_images_for_training(name)
    if images:
        label_images(images, name)
        with open(student_details_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, course, datetime.datetime.now()])
        messagebox.showinfo("Success", "Student profile saved successfully!")
        train_recognizer()  # Train the recognizer after saving profile

# Function to save teacher profile
def save_teacher_profile():
    name = teacher_name.get()
    department = teacher_department.get()
    if not name or not department:
        messagebox.showerror("Error", "Please enter both Name and Department")
        return

    images = capture_images_for_training(name)
    if images:
        label_images(images, name)
        with open(teacher_details_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, department, datetime.datetime.now()])
        messagebox.showinfo("Success", "Teacher profile saved successfully!")
        train_recognizer()  # Train the recognizer after saving profile

# Function to train the recognizer
def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []
    label_id = 0
    labels = {}

    for root_dir, dirs, files in os.walk(training_images_and_labels_dir):
        for file in files:
            if file.endswith("jpg"):
                label = os.path.basename(root_dir)
                if label not in labels:
                    labels[label] = label_id
                    label_id += 1
                img_path = os.path.join(root_dir, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                ids.append(labels[label])

    if faces:
        recognizer.train(faces, np.array(ids))
        recognizer.save(trainer_file)
        with open(labels_file, 'wb') as f:
            pickle.dump(labels, f)
    else:
        messagebox.showwarning("Warning", "No training data found. Ensure you have saved some profiles with images.")

# Function to take student attendance
def take_student_attendance():
    if not os.path.exists(trainer_file) or not os.path.exists(labels_file):
        messagebox.showerror("Error", "Trainer or labels file not found. Ensure the recognizer is trained.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_file)
    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)
    reverse_labels = {v: k for k, v in labels.items()}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access the webcam.")
        return

    attendance_marked = False  # Flag to indicate if attendance is marked

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, confidence = recognizer.predict(roi_gray)
            if confidence >= 45 and confidence <= 85:
                name = reverse_labels[id_]
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(student_attendance_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, timestamp])
                messagebox.showinfo("Success", f"Attendance marked for {name}")
                attendance_marked = True
                break

        cv2.imshow('Taking Attendance', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or attendance_marked:
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to take teacher attendance
import time

def take_teacher_attendance():
    if not os.path.exists(trainer_file) or not os.path.exists(labels_file):
        messagebox.showerror("Error", "Trainer or labels file not found. Ensure the recognizer is trained.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_file)
    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)
    reverse_labels = {v: k for k, v in labels.items()}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access the webcam.")
        return

    attendance_marked = False  # Flag to indicate if attendance is marked

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, confidence = recognizer.predict(roi_gray)
            if confidence >= 45 and confidence <= 85:
                name = reverse_labels[id_]
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(teacher_attendance_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, timestamp])
                messagebox.showinfo("Success", f"Attendance marked for {name}")
                attendance_marked = True
                break

        cv2.imshow('Taking Attendance', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or attendance_marked:
            break

        # Add a small delay to reduce the number of frames processed per second
        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

    # Break out of the loop as soon as attendance is marked
    if attendance_marked:
        return

# Function to handle admin login
def login():
    user = user_var.get()
    password = pass_var.get()
    if user == "admin" and password == "admin123":
        admin_dashboard()
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

# Function to display admin dashboard
def admin_dashboard():
    clear_screen()
    tk.Label(root, text="Admin Dashboard", font=("Helvetica", 16)).pack()
    ttk.Button(root, text="View Student Attendance", command=lambda: view_attendance("Student")).pack(pady=10)
    ttk.Button(root, text="View Teacher Attendance", command=lambda: view_attendance("Teacher")).pack(pady=10)
    ttk.Button(root, text="Logout", command=logout).pack(pady=10)
    ttk.Button(root, text="Back to Main Menu", command=create_main_screen).pack(pady=10)

def view_attendance(attendance_type):
    # Create a new window for displaying attendance records
    attendance_window = tk.Toplevel()
    if attendance_type == "Student":
        attendance_window.title("Student Attendance Records")
        file_path = student_attendance_file
    elif attendance_type == "Teacher":
        attendance_window.title("Teacher Attendance Records")
        file_path = teacher_attendance_file

    # Create a Treeview widget
    tree = ttk.Treeview(attendance_window)

    # Define columns
    tree["columns"] = ("Name", "Timestamp")
    tree.column("#0", width=0, stretch=tk.NO)  # Hide the first column (tree's index column)
    tree.column("Name", anchor=tk.W, width=200)
    tree.column("Timestamp", anchor=tk.W, width=200)

    # Create headings
    tree.heading("#0", text="", anchor=tk.W)
    tree.heading("Name", text="Name", anchor=tk.W)
    tree.heading("Timestamp", text="Timestamp", anchor=tk.W)

    # Read attendance records based on selection
    attendance_records = read_attendance(file_path)
    if attendance_records:
        for record in attendance_records:
            tree.insert("", tk.END, values=(record[0], record[1]))
    else:
        tree.insert("", tk.END, values=("No records found", ""))

    # Pack the Treeview widget
    tree.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    ttk.Button(attendance_window, text="Back to Main Menu", command=create_main_screen).pack(pady=10)

def read_attendance(file_path):
    try:
        with open(file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            attendance_records = list(reader)
            return attendance_records
    except FileNotFoundError:
        return []

# Function to logout from admin dashboard
def logout():
    clear_screen()
    create_login_screen()

# Function to clear screen
def clear_screen():
    for widget in root.winfo_children():
        widget.destroy()

# Function to create login screen
def create_login_screen():
    clear_screen()
    global user_var, pass_var
    user_var = tk.StringVar()
    pass_var = tk.StringVar()

    tk.Label(root, text="Username:", font=("Helvetica", 12)).pack()
    tk.Entry(root, textvariable=user_var, width=30).pack()

    tk.Label(root, text="Password:", font=("Helvetica", 12)).pack()
    tk.Entry(root, textvariable=pass_var, show="*", width=30).pack()

    tk.Button(root, text="Login", command=login, width=10).pack(pady=10)
    tk.Button(root, text="Back to Main Menu", command=create_main_screen).pack(pady=10)

# Function to create the main screen with buttons in the center
def create_main_screen():
    clear_screen()
    
    main_frame = ttk.Frame(root)
    main_frame.pack(expand=True)

    ttk.Button(main_frame, text="New Registration", command=new_registration_screen, width=30).pack(pady=10)
    ttk.Button(main_frame, text="Take Attendance", command=take_attendance_screen, width=30).pack(pady=10)
    ttk.Button(main_frame, text="Admin Login", command=create_login_screen, width=30).pack(pady=10)
    ttk.Button(main_frame, text="Quit", command=root.quit, width=30).pack(pady=10)

# Function to create new registration screen
def new_registration_screen():
    clear_screen()

    ttk.Label(root, text="Select Registration Type", font=("Helvetica", 16)).pack(pady=20)
    ttk.Button(root, text="Student Registration", command=student_registration_screen).pack(pady=10)
    ttk.Button(root, text="Teacher Registration", command=teacher_registration_screen).pack(pady=10)
    ttk.Button(root, text="Back to Main Menu", command=create_main_screen).pack(pady=10)

# Function to create student registration screen
def student_registration_screen():
    clear_screen()

    student_frame = ttk.LabelFrame(root, text="New Student Registration")
    student_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    ttk.Label(student_frame, text="Enter Name:", font=("Helvetica", 12)).grid(row=0, column=0, padx=10, pady=5, sticky="w")
    global student_name
    student_name = ttk.Entry(student_frame, width=30)
    student_name.grid(row=0, column=1, padx=10, pady=5)

    ttk.Label(student_frame, text="Enter Course:", font=("Helvetica", 12)).grid(row=1, column=0, padx=10, pady=5, sticky="w")
    global student_course
    student_course = ttk.Entry(student_frame, width=30)
    student_course.grid(row=1, column=1, padx=10, pady=5)

    ttk.Button(student_frame, text="Take Image via Webcam", command=save_student_profile).grid(row=2, column=0, columnspan=2, pady=10)
    ttk.Button(student_frame, text="Back to Registration Menu", command=new_registration_screen).grid(row=3, column=0, columnspan=2, pady=10)
    ttk.Button(student_frame, text="Back to Main Menu", command=create_main_screen).grid(row=4, column=0, columnspan=2, pady=10)

# Function to create teacher registration screen
def teacher_registration_screen():
    clear_screen()

    teacher_frame = ttk.LabelFrame(root, text="New Teacher Registration")
    teacher_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    ttk.Label(teacher_frame, text="Enter Name:", font=("Helvetica", 12)).grid(row=0, column=0, padx=10, pady=5, sticky="w")
    global teacher_name
    teacher_name = ttk.Entry(teacher_frame, width=30)
    teacher_name.grid(row=0, column=1, padx=10, pady=5)

    ttk.Label(teacher_frame, text="Enter Department:", font=("Helvetica", 12)).grid(row=1, column=0, padx=10, pady=5, sticky="w")
    global teacher_department
    teacher_department = ttk.Entry(teacher_frame, width=30)
    teacher_department.grid(row=1, column=1, padx=10, pady=5)

    ttk.Button(teacher_frame, text="Take Image via Webcam", command=save_teacher_profile).grid(row=2, column=0, columnspan=2, pady=10)
    ttk.Button(teacher_frame, text="Back to Registration Menu", command=new_registration_screen).grid(row=3, column=0, columnspan=2, pady=10)
    ttk.Button(teacher_frame, text="Back to Main Menu", command=create_main_screen).grid(row=4, column=0, columnspan=2, pady=10)

# Function to create take attendance screen
def take_attendance_screen():
    clear_screen()

    ttk.Label(root, text="Select Attendance Type", font=("Helvetica", 16)).pack(pady=20)
    ttk.Button(root, text="Take Student Attendance", command=take_student_attendance).pack(pady=10)
    ttk.Button(root, text="Take Teacher Attendance", command=take_teacher_attendance).pack(pady=10)
    ttk.Button(root, text="Back to Main Menu", command=create_main_screen).pack(pady=10)

# Start the application with the main screen
create_main_screen()

# Start GUI
root.mainloop()
