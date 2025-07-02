# Customer Insights Platform – Phân cụm dữ liệu bằng Streamlit

## 📌 Giới thiệu

Dự án này cung cấp một ứng dụng **Streamlit** giúp trực quan hóa và phân tích phân cụm dữ liệu (Clustering) dành cho các bài toán phân tích khách hàng, marketing, v.v.  
Bạn có thể upload dataset của mình, chọn thuật toán phân cụm, điều chỉnh tham số và xem trực quan kết quả dễ dàng.

---

## 🚀 Hướng dẫn cài đặt & chạy dự án

### 1. **Clone dự án về máy**

git clone https://github.com/dieplai/-Ai-applications-for-business.git
cd -Ai-applications-for-business

2. Tạo môi trường ảo (khuyến nghị)

python -m venv venv
# Kích hoạt (Windows)
venv\Scripts\activate
# Kích hoạt (Linux/Mac)
source venv/bin/activate

3. Cài đặt thư viện cần thiết

pip install -r requirements.txt

    ⚠️ Lưu ý: Nếu thiếu thư viện nào, kiểm tra lại file requirements.txt hoặc cài bổ sung bằng pip install <ten-thu-vien>

4. Chạy ứng dụng Streamlit

streamlit run app.py

    app.py là file chính. Nếu bạn dùng tên file khác, hãy đổi lại cho đúng.

5. Truy cập giao diện web

    Sau khi chạy thành công, truy cập địa chỉ mà Streamlit cung cấp (thường là: http://localhost:8501)
