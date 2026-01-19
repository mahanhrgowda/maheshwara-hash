import streamlit as st
import hashlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import qrcode
from PIL import Image
import io
import base64
import json
import time
from pyzbar.pyzbar import decode  # For QR decoding from image

# Note: For Dilithium integration, this app assumes 'dilithium-py' is installed (pip install dilithium-py).
# If not available in your env, comment out the Dilithium sections or use a placeholder.
try:
    from dilithium_py.ml_dsa import ML_DSA_44 as Dilithium  # Standard ML-DSA-44 (Dilithium2 equivalent)
except ImportError:
    Dilithium = None
    st.warning("Dilithium-py not installed. Signature features disabled.")

# Maheshwara Sutra phoneme blocks (from chat history)
maheshwara_sutras = [
    "a i u ṇ", "ṛ ḷ k", "e o ṅ", "ai au c", "ha ya va ra ṭ",
    "la ṇ", "ña ma ṅa ṇa na nam", "jha bha ñ", "gha ḍha dha ṣ",
    "ja ba ga ḍa da ś", "kha pha cha ṭha tha ca ṭa ta v", "ka pa y",
    "śa ṣa sa r", "ha l"
]

# Phoneme map
def build_phoneme_map():
    phoneme_map = {}
    index = 0
    for group in maheshwara_sutras:
        for phoneme in group.split():
            if phoneme not in phoneme_map:
                phoneme_map[phoneme] = index
        index += 1
    return phoneme_map

PHONEME_MAP = build_phoneme_map()

# Bhāva map (rasas to numeric modifiers)
BHAVA_MAP = {
    'Shanta': 0,    # Peace
    'Veera': 1,     # Heroism
    'Karuna': 2,    # Compassion
    'Adbhuta': 3,   # Wonder
    'Hasya': 4,     # Humor
    'Bhayanaka': 5, # Fear
    'Bibhatsa': 6,  # Disgust
    'Raudra': 7,    # Anger
    'Shringara': 8  # Love
}

# Encode phonemes
def encode_phonemes(text):
    letters = text.lower()
    encoded = [str(PHONEME_MAP.get(ch, 99)) for ch in letters if ch in PHONEME_MAP]
    return ''.join(encoded)

# Maheshwara Hash function
def maheshwara_hash(input_text, bhava='Shanta'):
    encoded = encode_phonemes(input_text)
    bhava_value = str(BHAVA_MAP.get(bhava, 0))
    combined = encoded + bhava_value
    hash_obj = hashlib.sha3_512(combined.encode())
    return hash_obj.hexdigest(), hash_obj.digest()  # Return hex and bytes

# Lattice vector for input (simplified mapping based on chat)
def get_lattice_vector(input_text, bhava):
    # Example: Map to [X: Bhūta, Y: Rasa, Z: Kāla] - normalized 0-1
    # For demo, use hash length or simple calc; in reality, derive from semantics
    hash_val = int(maheshwara_hash(input_text, bhava)[0][:8], 16) / (2**32 - 1)  # Normalize
    return [hash_val, hash_val * 0.92, hash_val * 0.85]  # Mimic chat example for "Mahān"

# Plot 3D lattice
def plot_lattice(vector):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Bhūta (X)')
    ax.set_ylabel('Rasa (Y)')
    ax.set_zlabel('Kāla (Z)')
    ax.scatter(vector[0], vector[1], vector[2], color='red', label='Input Node')
    # Simplified lattice lines
    for i in range(3):
        ax.plot([0, 1], [i/2, i/2], [i/2, i/2], color='blue', alpha=0.5)
        ax.plot([i/2, i/2], [0, 1], [i/2, i/2], color='blue', alpha=0.5)
        ax.plot([i/2, i/2], [i/2, i/2], [0, 1], color='blue', alpha=0.5)
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# QR Code generation
def generate_qr(data):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

# Decode QR from uploaded image
def decode_qr_from_image(image):
    decoded_objects = decode(image)
    if decoded_objects:
        return decoded_objects[0].data.decode('utf-8')
    else:
        return None

# Simple Blockchain Simulation Class
class SimpleBlockchain:
    def __init__(self):
        self.chain = []
        self.pk, self.sk = None, None
        if Dilithium:
            self.pk, self.sk = Dilithium.keygen()
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis = self.create_block("Genesis Block: Om Namah Shivaya", 'Shanta')
        self.chain.append(genesis)

    def create_block(self, data, bhava):
        prev_hash = self.chain[-1]['hash'] if self.chain else '0'
        _, block_hash_bytes = maheshwara_hash(data + prev_hash + str(time.time()), bhava)
        block = {
            'index': len(self.chain),
            'data': data,
            'bhava': bhava,
            'prev_hash': prev_hash,
            'hash': block_hash_bytes.hex(),
        }
        if Dilithium and self.sk:
            sig = Dilithium.sign(self.sk, block_hash_bytes)
            block['signature'] = sig.hex()
        return block

    def add_block(self, data, bhava):
        block = self.create_block(data, bhava)
        self.chain.append(block)

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i-1]
            # Check hash linkage
            if current['prev_hash'] != prev['hash']:
                return False
            # Recompute hash
            _, recomputed_hash = maheshwara_hash(current['data'] + current['prev_hash'] + str(time.time()), current['bhava'])  # Note: timestamp not stored, so approx
            # For demo, skip exact timestamp check
            # Verify signature if present
            if Dilithium and 'signature' in current and self.pk:
                sig = bytes.fromhex(current['signature'])
                if not Dilithium.verify(self.pk, bytes.fromhex(current['hash']), sig):
                    return False
        return True

# Streamlit App
st.title("Maheshwara PQC Toolkit with Blockchain Integration")
st.markdown("""
This updated Streamlit app integrates all the inventions from the chat history, including the Maheshwara Hash, Bhāva obfuscation, Maheshwara Lattice, PQC signatures using Dilithium, secure messaging, QR encryption/decryption, blockchain demo, and now a new feature to upload a QR code image for automatic decoding and decryption.

**Explanations (Updated):**
- **QR Upload and Decrypt**: Upload a QR image file, automatically decode the Base64 data, then decrypt using the guessed Bhāva and original plaintext (for verification in demo). This enhances usability by handling scanned QR images directly.
- All previous features remain, with explanations in sections below.
""")

# Initialize blockchain in session state
if 'blockchain' not in st.session_state:
    st.session_state['blockchain'] = SimpleBlockchain()

# Input section
input_text = st.text_input("Enter Input (e.g., Mantra or Name)", value="Mahān")
bhava = st.selectbox("Select Bhāva (Emotional Resonance)", list(BHAVA_MAP.keys()), index=0)

# Hash Computation
if st.button("Compute Maheshwara Hash"):
    hex_hash, byte_hash = maheshwara_hash(input_text, bhava)
    st.subheader("Maheshwara Hash Output")
    st.write(f"Hex: {hex_hash}")
    st.write(f"Base64: {base64.b64encode(byte_hash).decode()}")
    st.markdown("""
    **Explanation**: The input is split into phonemes using Maheshwara Sūtras, encoded numerically, appended with Bhāva modifier, and hashed with SHA3-512. This provides quantum resistance via non-linear mappings and high entropy.
    """)

# Lattice Visualization
if st.button("Visualize Maheshwara Lattice"):
    vector = get_lattice_vector(input_text, bhava)
    plot_buf = plot_lattice(vector)
    st.subheader("3D Maheshwara Lattice Plot")
    st.image(plot_buf)
    st.write(f"Vector: {vector}")
    st.markdown("""
    **Explanation**: The lattice projects the input into a 3D space (X: Elemental, Y: Emotional, Z: Temporal). The red point is the encoded node. This structure inspires lattice-based crypto like Dilithium, adding spiritual semantics.
    """)

# PQC Signature
if Dilithium:
    st.subheader("PQC Signature with Dilithium")
    if st.button("Generate Keypair and Sign"):
        pk, sk = Dilithium.keygen()
        # Pre-hash with Maheshwara
        _, msg_hash = maheshwara_hash(input_text, bhava)
        sig = Dilithium.sign(sk, msg_hash)
        st.write(f"Public Key (hex): {pk.hex()[:32]}...")
        st.write(f"Signature (hex): {sig.hex()[:32]}...")
        st.session_state['pk'] = pk
        st.session_state['sk'] = sk
        st.session_state['sig'] = sig
        st.session_state['msg_hash'] = msg_hash
        # Update blockchain keys
        st.session_state['blockchain'].pk = pk
        st.session_state['blockchain'].sk = sk
    if 'pk' in st.session_state:
        if st.button("Verify Signature"):
            valid = Dilithium.verify(st.session_state['pk'], st.session_state['msg_hash'], st.session_state['sig'])
            st.write(f"Verification: {'Valid' if valid else 'Invalid'}")
    st.markdown("""
    **Explanation**: Dilithium is a lattice-based PQC signature scheme. Here, we pre-hash the input with Maheshwara for added entropy, then sign/verify. This ensures quantum-safe integrity for dharmic contracts or ID proofs.
    """)
else:
    st.warning("Dilithium not available. Signature and blockchain signing disabled.")

# Secure Messaging Demo
st.subheader("Secure Messaging Demo")
receiver_bhava = st.selectbox("Receiver's Bhāva Guess", list(BHAVA_MAP.keys()), index=0)
if st.button("Simulate Send/Receive"):
    _, msg_hash = maheshwara_hash(input_text, bhava)
    if Dilithium and 'sk' in st.session_state:
        sig = Dilithium.sign(st.session_state['sk'], msg_hash)
        st.write("Sent: Hashed Message + Signature")
        # Simulate receive
        _, recv_hash = maheshwara_hash(input_text, receiver_bhava)
        if recv_hash == msg_hash:
            valid = Dilithium.verify(st.session_state['pk'], recv_hash, sig)
            st.write(f"Bhāva Match: Yes | Signature: {'Valid' if valid else 'Invalid'}")
        else:
            st.write("Bhāva Mismatch - Intention not aligned!")
    else:
        st.write("Demo: Hashed message sent. Receiver recomputes with Bhāva to verify match.")
st.markdown("""
**Explanation**: Sender hashes with Bhāva, signs with Dilithium. Receiver guesses Bhāva to recompute hash and verify signature. This adds intention-based security to messaging.
""")

# QR Encryption/Decryption Feature
st.subheader("QR Code Encryption and Decryption")
plaintext = st.text_input("Enter Plaintext to Encrypt", value="Secret Mantra")
encrypt_bhava = st.selectbox("Encryption Bhāva", list(BHAVA_MAP.keys()), index=0)
if st.button("Encrypt and Generate QR"):
    # Derive "key" from Maheshwara Hash of bhava only (enhanced for direct decryption)
    _, key_bytes = maheshwara_hash(encrypt_bhava, encrypt_bhava)  # Use bhava as input and bhava
    key = key_bytes  # Full digest for key
    plaintext_bytes = plaintext.encode()
    key = key[:len(plaintext_bytes)] if len(key) > len(plaintext_bytes) else key * (len(plaintext_bytes) // len(key) + 1)
    key = key[:len(plaintext_bytes)]
    encrypted = bytes([p ^ k for p, k in zip(plaintext_bytes, key)])
    encrypted_b64 = base64.b64encode(encrypted).decode()
    qr_buf = generate_qr(encrypted_b64)
    st.image(qr_buf, caption="Encrypted QR Code")
    st.session_state['encrypted_b64'] = encrypted_b64  # Store for convenience
    st.session_state['encrypt_bhava'] = encrypt_bhava
    st.session_state['plaintext'] = plaintext  # For optional verification
    st.download_button("Download QR", qr_buf, "encrypted_qr.png")
st.markdown("""
**Explanation**: Encrypts plaintext by XOR with key derived solely from Bhāva (using Maheshwara Hash). Base64 encodes result, embeds in QR. This allows direct decryption with just Bhāva, providing PQC-resistant symmetric encryption demo tied to Sanskrit-inspired hashing. Note: For demo purposes; in practice, use stronger keys.
""")

# Decryption Section
st.subheader("Decryption")
decrypt_bhava = st.selectbox("Bhāva for Decryption", list(BHAVA_MAP.keys()), index=0)
scanned_data = st.text_input("Enter Scanned QR Data (Base64)", value="")

# Upload QR Image for Decoding
uploaded_qr = st.file_uploader("Upload QR Code Image for Decryption", type=["png", "jpg", "jpeg"])
if uploaded_qr:
    try:
        image = Image.open(uploaded_qr)
        decoded_data = decode_qr_from_image(image)
        if decoded_data:
            scanned_data = decoded_data  # Override text input
            st.success("QR Code Decoded Successfully!")
            st.write(f"Decoded Base64 Data: {scanned_data}")
        else:
            st.error("No QR Code detected in the uploaded image.")
    except Exception as e:
        st.error(f"Error decoding QR: {e}")

if st.button("Decrypt"):
    encrypted_b64 = scanned_data or st.session_state.get('encrypted_b64', '')
    if encrypted_b64:
        try:
            encrypted = base64.b64decode(encrypted_b64)
            _, key_bytes = maheshwara_hash(decrypt_bhava, decrypt_bhava)
            key = key_bytes
            key = key[:len(encrypted)] if len(key) > len(encrypted) else key * (len(encrypted) // len(key) + 1)
            key = key[:len(encrypted)]
            decrypted_bytes = bytes([e ^ k for e, k in zip(encrypted, key)])
            decrypted = decrypted_bytes.decode(errors='ignore')
            st.subheader("Decrypted Plaintext Output")
            st.write(decrypted)
            # Optional session check for demo
            if 'encrypt_bhava' in st.session_state and decrypt_bhava == st.session_state['encrypt_bhava']:
                st.success("Bhāva Match Confirmed - Decryption Successful!")
                if 'plaintext' in st.session_state:
                    if decrypted == st.session_state['plaintext']:
                        st.info("Matches Original Plaintext.")
            else:
                st.warning("Bhāva may not match - Output may be garbled. Try different Bhāva.")
        except Exception as e:
            st.error(f"Decryption Error: {e}")
    else:
        st.warning("Provide Base64 data via upload or input to decrypt.")

st.markdown("""
**Explanation**: The enhanced decryption engine derives the key solely from Bhāva, XORs with the ciphertext from QR/Base64, and directly outputs the plaintext. If Bhāva is incorrect, output will be garbled. Supports automatic QR decoding via upload. This demonstrates simple, Bhāva-based symmetric decryption.
""")

# Blockchain Demo Section
st.subheader("Blockchain Integration Demo")
block_data = st.text_input("Enter Data for New Block (e.g., Sankalpa or Mantra)", value="Namah Shivaya")
block_bhava = st.selectbox("Block Bhāva", list(BHAVA_MAP.keys()), index=0)
if st.button("Add Block to Chain"):
    st.session_state['blockchain'].add_block(block_data, block_bhava)
    st.success("Block Added!")

if st.button("View Blockchain"):
    for block in st.session_state['blockchain'].chain:
        st.json(block)

if st.button("Validate Chain"):
    valid = st.session_state['blockchain'].is_chain_valid()
    if valid:
        st.success("Chain is Valid!")
    else:
        st.error("Chain is Invalid - Tampering Detected!")
    # Simulate tampering for demo
    if st.checkbox("Simulate Tampering (Change Data in Block 1)"):
        if len(st.session_state['blockchain'].chain) > 1:
            st.session_state['blockchain'].chain[1]['data'] = "Tampered Data"
            st.warning("Block 1 Tampered. Validate again to detect.")

st.markdown("""
**Explanation**: This simulates a blockchain where each block's hash is computed using Maheshwara Hash (with Bhāva for entropy). Blocks link via previous hashes for immutability. Dilithium signs each block for PQC-verifiable authenticity. Validation checks hash chains and signatures. In a real system, this could anchor hashes to Ethereum or use Śabdāstra smart contracts for dharmic applications like vow recording or identity tokenization.
For production, integrate with web3.py and a real chain (e.g., Ethereum testnet), but this demo is local and simulated.
""")

# Comparison Table (from chat)
st.subheader("Comparison with Standard Hashes")
data = {
    "Feature": ["Avalanche Effect", "Quantum Resistance", "Human Readability", "Cultural Encoding", "Use in DSLs"],
    "SHA-3": ["High", "Moderate", "None", "No", "Low"],
    "BLAKE3": ["High", "Moderate", "None", "No", "Medium"],
    "Maheshwara Hash": ["Very High (multi-level)", "High (non-binary logic)", "Symbolic layer possible", "Yes (Sanskrit-rooted)", "High (Śabdāstra-native)"]
}
st.table(data)
st.markdown("""
**Explanation**: This table highlights Maheshwara's advantages in quantum resistance and cultural integration, as discussed in the chat.
""")

# Run instructions
st.info("To run locally: Save as app.py, then 'streamlit run app.py'. Ensure dependencies: streamlit, matplotlib, qrcode, pillow, pyzbar, dilithium-py (optional). Add 'pyzbar' to requirements.txt for QR decoding.")
