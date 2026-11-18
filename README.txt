راهنمای اجرای سامانه پیش‌بینی زمان تحویل (Lead Time Prediction Platform)
========================================================================

پیش‌نیازها
----------
1) ویندوز ۱۰ یا ۱۱ (۶۴ بیتی)
2) پایتون نسخه ۳.۱۰ یا جدیدتر نصب باشد:
   https://www.python.org/downloads/
   در اولین صفحه نصب، گزینه «Add python.exe to PATH» را حتماً تیک بزنید.
3) فقط در اولین اجرا به اینترنت نیاز است تا کتابخانه‌های موردنیاز نصب شوند.

نحوه اجرا
---------
1) فایل ZIP را در یک مسیر دلخواه استخراج کنید (ساختار پوشه‌ها را کامل نگه دارید).
2) روی فایل  run_platform.bat  دوبار کلیک کنید.
3) اسکریپت به‌طور خودکار:
   - پایتون را پیدا می‌کند،
   - کتابخانه‌های لازم را (در صورت نیاز) نصب می‌کند،
   - سرویس را روی پورت 8001 اجرا می‌کند،
   - و داشبورد را در مرورگر پیش‌فرض باز می‌کند.

نکات مهم
--------
- اجرای دوباره این فایل بی‌خطر است و سرویس تکراری ساخته نمی‌شود (سرویس فعال دوباره استفاده می‌شود).
- سرویس در پس‌زمینه اجرا می‌شود و پس از بستن پنجره هم فعال می‌ماند.
- برای توقف سرویس، دستور زیر را اجرا کنید (شماره PID در پنجره اسکریپت و در فایل logs\backend.pid نوشته می‌شود):
      taskkill /PID <شماره_PID> /F
- گزارش رویدادها و خطاها در فایل  logs\backend.log  ذخیره می‌شود.

آدرس‌ها
-------
- داشبورد: app\frontend\index.html  (به‌صورت خودکار باز می‌شود)
- مستندات سرویس: http://127.0.0.1:8001/docs
- وضعیت سرویس: http://127.0.0.1:8001/health


---------------- English quick start ----------------
Requirements
  1) Windows 10/11 (64-bit)
  2) Python 3.10 or newer. During installation tick "Add python.exe to PATH".
     Download: https://www.python.org/downloads/
  3) Internet access is needed on the FIRST run only (to install packages).

Run
  1) Extract the whole ZIP to any folder (keep the folder structure intact).
  2) Double-click  run_platform.bat
  3) The script automatically finds Python, installs the required packages if
     needed, starts the backend on port 8001, and opens the dashboard in your
     default browser.

Notes
  - Running it again is safe: it reuses the already running server.
  - The server keeps running in the background after you close the window.
  - Stop it with:  taskkill /PID <pid> /F   (the PID is shown in the window and
    saved to logs\backend.pid)
  - Logs are written to  logs\backend.log
  - Service docs: http://127.0.0.1:8001/docs    Health: http://127.0.0.1:8001/health
