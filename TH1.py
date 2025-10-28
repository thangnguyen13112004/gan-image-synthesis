import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageDraw
from io import BytesIO
import tempfile
from streamlit_drawable_canvas import st_canvas

# Cấu hình trang
st.set_page_config(
    page_title="AI Model Demo",
    page_icon="🤖",
    layout="wide"
)

# Tiêu đề chính
st.title("🤖 AI Model Demo App")
st.markdown("### Chọn model bạn muốn sử dụng:")

# Cache để load models (tránh reload liên tục)
@st.cache_resource
def load_inpainting_model():
    try:
        model = load_model('Models/Inpainting_epoch_741.h5')
        return model
    except Exception as e:
        st.error(f"Không thể load Inpainting model: {e}")
        return None

@st.cache_resource
def load_edge2shoes_model():
    try:
        model = load_model('Models/Edge2shoes_epoch_341.h5')
        return model
    except Exception as e:
        st.error(f"Không thể load Edge2Shoes model: {e}")
        return None

# === INPAINTING FUNCTIONS ===
def preprocess_inpaint_image(img, size=(256, 256)):
    """Tiền xử lý ảnh cho Inpainting model"""
    img = img.resize(size).convert('RGB')
    arr = np.array(img).astype(np.float32)
    arr = (arr - 127.5) / 127.5
    return np.expand_dims(arr, axis=0)

def postprocess_inpaint_image(img_array):
    """Hậu xử lý ảnh từ Inpainting model"""
    img_array = (img_array[0] + 1) / 2.0
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def create_masked_image(original_image, canvas_data):
    """Tạo ảnh mới với vùng đen từ canvas"""
    if canvas_data is None or canvas_data.image_data is None:
        return original_image
    
    # Chuyển ảnh gốc về kích thước 256x256
    original_resized = original_image.resize((256, 256)).convert('RGB')
    original_array = np.array(original_resized)
    
    # Lấy dữ liệu từ canvas
    canvas_array = canvas_data.image_data.astype(np.uint8)
    
    # Tạo mask từ canvas (vùng được vẽ)
    if len(canvas_array.shape) == 3:
        # Kiểm tra xem có alpha channel không
        if canvas_array.shape[2] == 4:
            # Sử dụng alpha channel để tạo mask
            mask = canvas_array[:, :, 3] > 0
        else:
            # Chuyển RGB sang grayscale để tạo mask
            gray = cv2.cvtColor(canvas_array[:, :, :3], cv2.COLOR_RGB2GRAY)
            mask = gray > 10
    else:
        mask = canvas_array > 10
    
    # Tạo ảnh mới: giữ nguyên ảnh gốc, chỉ tô đen vùng được vẽ
    masked_image = original_array.copy()
    
    # Tô đen vùng được vẽ
    if len(mask.shape) == 2:
        masked_image[mask] = [0, 0, 0]  # Tô đen
    
    return Image.fromarray(masked_image)

# === EDGE2SHOES FUNCTIONS ===
def show_edge2shoes_images(input_img, predicted_img):
    """Hiển thị ảnh cho Edge2Shoes"""
    def to_display(img):
        # Nếu ảnh có giá trị âm (đang ở [-1, 1]), thì chuyển về [0, 1]
        if np.min(img) < 0:
            img = (img + 1) / 2.0
        return np.clip(img, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Input image (Edge)
    axes[0].imshow(to_display(input_img[0]))
    axes[0].set_title("Input (Edge)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Predicted image (Shoes)
    axes[1].imshow(to_display(predicted_img[0]))
    axes[1].set_title("Generated (Shoes)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

def load_and_preprocess_edge_image(uploaded_file, size=(256, 256)):
    """Xử lý ảnh input cho Edge2Shoes"""
    try:
        # Đọc ảnh từ uploaded file
        image = Image.open(uploaded_file)
        
        # Resize ảnh
        image = image.resize(size)
        
        # Convert sang RGB nếu cần
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert sang array và normalize
        img_array = img_to_array(image)
        img_array = (img_array - 127.5) / 127.5  # Normalize to [-1, 1]
        
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh: {e}")
        return None

# Tạo tab bar
tab1, tab2 = st.tabs(["🎨 Inpainting Model", "👟 Edge2Shoes Model"])

# === TAB 1: INPAINTING MODEL ===
with tab1:
    st.header("🎨 Inpainting Model")
    st.info("🖌️ Tải ảnh và vẽ phần muốn xóa (tô đen), sau đó nhấn nút để xử lý.")
    
    # Load model
    inpaint_model = load_inpainting_model()
    
    if inpaint_model is not None:
        # Upload ảnh
        uploaded_file = st.file_uploader(
            "Chọn ảnh cần inpainting:",
            type=["jpg", "png", "jpeg"],
            key="inpaint_upload"
        )
        
        if uploaded_file:
            original_image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Ảnh Gốc")
                st.image(original_image, caption="Ảnh gốc", use_column_width=True)
            
            with col2:
                st.subheader("🎨 Cài Đặt Vẽ")
                drawing_mode = st.selectbox(
                    "Chế độ vẽ:",
                    ["freedraw", "rect", "circle"],
                    key="inpaint_drawing_mode"
                )
                stroke_width = st.slider(
                    "Độ dày nét:",
                    5, 100, 20,
                    key="inpaint_stroke_width"
                )
            
                # Canvas vẽ với ảnh gốc làm nền
                st.subheader("🖌️ Vẽ phần muốn xóa (tô đen)")
                st.markdown("*Vẽ lên những vùng bạn muốn AI điền vào*")
                
                # Resize ảnh để hiển thị trong canvas
                display_image = original_image.resize((256, 256))
                
                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 1.0)" if drawing_mode in ["rect", "circle"] else "rgba(0, 0, 0, 0)",
                    stroke_width=stroke_width,
                    stroke_color="#000000",
                    background_image=display_image,
                    update_streamlit=True,
                    height=256,
                    width=256,
                    drawing_mode=drawing_mode,
                    key="inpaint_canvas"
                )
                
            # Nút xử lý
            if st.button("🧩 Chạy Inpainting", type="primary", key="run_inpainting"):
                if canvas_result.image_data is not None:
                    with st.spinner("🎨 Đang xử lý inpainting..."):
                        try:
                            # Tạo ảnh có vùng đen
                            masked_image = create_masked_image(original_image, canvas_result)
                            
                            # Tiền xử lý cho model
                            input_arr = preprocess_inpaint_image(masked_image)
                            
                            # Dự đoán
                            predicted = inpaint_model.predict(input_arr, verbose=0)
                            
                            # Hậu xử lý
                            predicted_img = postprocess_inpaint_image(predicted)
                            
                            # Hiển thị kết quả
                            st.subheader("📊 Kết Quả Inpainting")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.image(
                                    original_image.resize((256, 256)), 
                                    caption="Ảnh Gốc", 
                                    use_column_width=True
                                )
                            
                            with col2:
                                st.image(
                                    masked_image, 
                                    caption="Ảnh Input (vùng đen)", 
                                    use_column_width=True
                                )
                            
                            with col3:
                                st.image(
                                    predicted_img, 
                                    caption="Kết Quả Inpainting", 
                                    use_column_width=True
                                )
                            
                            st.success("✅ Inpainting hoàn thành!")
                            
                            # Nút download
                            st.subheader("💾 Tải Xuống")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Download ảnh input (có vùng đen)
                                input_buffer = BytesIO()
                                masked_image.save(input_buffer, format='PNG')
                                st.download_button(
                                    label="📥 Tải ảnh Input",
                                    data=input_buffer.getvalue(),
                                    file_name="input_masked.png",
                                    mime="image/png",
                                    key="download_input"
                                )
                            
                            with col2:
                                # Download kết quả
                                result_buffer = BytesIO()
                                predicted_img.save(result_buffer, format='PNG')
                                st.download_button(
                                    label="📥 Tải kết quả",
                                    data=result_buffer.getvalue(),
                                    file_name="result_inpainted.png",
                                    mime="image/png",
                                    key="download_result"
                                )
                                
                        except Exception as e:
                            st.error(f"❌ Lỗi khi xử lý: {str(e)}")
                else:
                    st.warning("⚠️ Hãy vẽ lên ảnh trước khi xử lý!")
        else:
            st.info("👆 Vui lòng tải ảnh lên để bắt đầu")
            
            # Hướng dẫn sử dụng
            with st.expander("📖 Hướng dẫn sử dụng Inpainting"):
                st.markdown("""
                **Các bước thực hiện:**
                1. 📤 **Upload ảnh** cần xử lý
                2. 🎨 **Chọn công cụ vẽ**: Freedraw (vẽ tự do), Rect (hình chữ nhật), Circle (hình tròn)
                3. 🖌️ **Vẽ lên vùng** bạn muốn AI điền vào (tô đen)
                4. 🚀 **Nhấn "Chạy Inpainting"** để xử lý
                5. 💾 **Tải xuống** kết quả
                
                **Lưu ý:**
                - Vẽ chính xác vùng cần xóa/thay thế
                - Ảnh sẽ được resize về 256x256 để xử lý
                - Kết quả tốt nhất với các vùng không quá phức tạp
                """)
    else:
        st.error("❌ Không thể load Inpainting model!")

# === TAB 2: EDGE2SHOES MODEL ===
with tab2:
    st.header("👟 Edge2Shoes Model")
    st.info("🔄 Chuyển đổi từ edge/sketch thành hình ảnh giày thật")
    
    # Load model
    edge2shoes_model = load_edge2shoes_model()
    
    if edge2shoes_model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Upload Ảnh Edge")
            edge_file = st.file_uploader(
                "Chọn ảnh edge/sketch của giày:",
                type=['png', 'jpg', 'jpeg'],
                key="edge_upload",
                help="Upload ảnh edge hoặc sketch của giày để tạo ra hình ảnh giày thật"
            )
            
            if edge_file:
                # Hiển thị ảnh gốc
                st.image(edge_file, caption="Ảnh Edge Input", use_column_width=200)
                
                # Nút để chạy model
                if st.button("👟 Vẽ Giày", type="primary", key="generate_shoes"):
                    with st.spinner("🎨 Đang tạo giày..."):
                        try:
                            # Xử lý ảnh input
                            input_img = load_and_preprocess_edge_image(edge_file)
                            
                            if input_img is not None:
                                # Dự đoán
                                predicted = edge2shoes_model.predict(input_img, verbose=0)
                                
                                # Hiển thị kết quả trong cột 2
                                with col2:
                                    st.subheader("🎯 Kết Quả")
                                    
                                    # Tạo figure để hiển thị
                                    fig = show_edge2shoes_images(input_img, predicted)
                                    st.pyplot(fig)
                                    
                                    st.success("✅ Tạo giày thành công!")
                                    
                                    # Thêm thông tin
                                    with st.expander("📊 Thông tin chi tiết"):
                                        st.write(f"**Input shape:** {input_img.shape}")
                                        st.write(f"**Output shape:** {predicted.shape}")
                                        st.write(f"**Model:** g_model_epoch_311.h5")
                            else:
                                st.error("❌ Không thể xử lý ảnh input!")
                                
                        except Exception as e:
                            st.error(f"❌ Lỗi khi tạo giày: {e}")
        
        with col2:
            if not edge_file:
                st.subheader("🎯 Kết Quả")
                st.info("👆 Hãy upload ảnh edge ở bên trái và nhấn 'Vẽ Giày'")
                
                # Hiển thị ví dụ hoặc hướng dẫn
                st.markdown("""
                **Hướng dẫn sử dụng:**
                1. 📤 Upload ảnh edge/sketch của giày
                2. 👟 Nhấn nút "Vẽ Giày"
                3. 🎉 Xem kết quả được tạo ra!
                
                **Lưu ý:**
                - Ảnh nên có kích thước vuông (256x256 là tối ưu)
                - Edge càng rõ ràng thì kết quả càng tốt
                - Hỗ trợ format: PNG, JPG, JPEG
                """)
    else:
        st.error("❌ Không thể load Edge2Shoes model!")

# Sidebar với thông tin
with st.sidebar:
    st.markdown("## 📝 Thông Tin")
    st.markdown("""
    ### 🤖 Models Available:
    - **Inpainting Model**: Điền vào vùng thiếu trong ảnh
    - **Edge2Shoes Model**: Tạo giày từ edge/sketch
    
    ### 📊 Model Info:
    - **Inpainting**: Inpainting_epoch_501.h5
    - **Edge2Shoes**: g_model_epoch_311.h5
    - **Input Size**: 256x256
    - **Framework**: TensorFlow/Keras
    """)
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Requirements:")
    st.code("""
    pip install streamlit
    pip install tensorflow
    pip install opencv-python
    pip install streamlit-drawable-canvas
    pip install pillow
    pip install matplotlib
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# CSS để làm đẹp
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: bold;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stSuccess {
        border-radius: 10px;
    }
    
    .stInfo {
        border-radius: 10px;
    }
    
    .stError {
        border-radius: 10px;
    }
    
    .stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)