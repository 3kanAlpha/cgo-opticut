import numpy as np
from PIL import Image
from skimage import color
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def process_image(img_rgb_pil: Image.Image, img_edit_pil: Image.Image):
  return process_image_lab(img_rgb_pil, img_edit_pil)

def process_image_lab(img_rgb_pil: Image.Image, img_edit_pil: Image.Image, sigma: float = 10.0):
    """
    CIELAB色空間を利用してコスト関数を改善した前景背景分離関数。

    Args:
        img_rgb_pil (Image.Image): 元のカラー画像。
        img_edit_pil (Image.Image): 前景・背景のヒントが書き込まれた編集画像。
        sigma (float): 色の類似性を測るためのグローバルなシグマ値。
    """
    img_rgb = np.array(img_rgb_pil.convert("RGB")) / 255.0
    img_gray = np.array(img_rgb_pil.convert("L")) / 255.0
    img_edit = np.array(img_edit_pil.convert("L")) / 255.0

    # ヒントマスクの作成
    img_hint = np.full(img_edit.shape, 0.5)
    idx = (np.abs(img_gray - img_edit) > 1e-4)
    img_hint[idx] = img_edit[idx]
    white_mask = img_hint > 0.99
    black_mask = img_hint < 0.01

    # CIELAB色空間に変換
    img_lab = color.rgb2lab(img_rgb)
    
    h, w, _ = img_lab.shape
    n_pixels = h * w
    
    row = []
    col = []
    dat = []
    b = np.zeros(n_pixels)

    # 近傍の座標
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # 制約項の重み(lambda)
    CONSTRAINT_WEIGHT = 100.0

    for u in range(h):
        for v in range(w):
            i = u * w + v
            
            # ユーザー指定のヒント（制約）がある場合
            if white_mask[u, v]:
                row.append(i)
                col.append(i)
                dat.append(CONSTRAINT_WEIGHT)
                b[i] = CONSTRAINT_WEIGHT
            elif black_mask[u, v]:
                row.append(i)
                col.append(i)
                dat.append(CONSTRAINT_WEIGHT)
                b[i] = 0
            else:
                # 制約がない場合、近傍ピクセルとの関係性を定義
                sum_weights = 0
                lab_i = img_lab[u, v]

                for du, dv in neighbors:
                    nu, nv = u + du, v + dv
                    if 0 <= nu < h and 0 <= nv < w:
                        j = nu * w + nv
                        lab_j = img_lab[nu, nv]
                        
                        # CIELAB空間でのユークリッド距離に基づく重み
                        dist_sq = np.sum((lab_i - lab_j) ** 2)
                        weight = np.exp(-dist_sq / (2 * sigma ** 2))
                        
                        row.append(i)
                        col.append(j)
                        dat.append(-weight)
                        sum_weights += weight
                
                # 対角成分
                row.append(i)
                col.append(i)
                dat.append(sum_weights)

    A = csr_matrix((dat, (row, col)), shape=(n_pixels, n_pixels))

    # 連立一次方程式を解く
    alpha = spsolve(A, b).reshape(h, w)
    
    # 結果をクリッピングして正規化
    alpha = np.clip(alpha, 0, 1)

    # 最終的な前景画像を作成
    img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)
    alpha_channel = (alpha * 255).astype(np.uint8)
    rgba_image = np.dstack((img_rgb_uint8, alpha_channel))
    
    foreground_image_pil = Image.fromarray(rgba_image, mode='RGBA')
    mask_binary_pil = Image.fromarray(alpha_channel, mode='L')

    return mask_binary_pil, foreground_image_pil