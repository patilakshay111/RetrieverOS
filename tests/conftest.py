import os


os.environ["DATABASE_URL"] = "sqlite:///./.pytest-runtime.db"
os.environ["SMTP_HOST"] = ""
