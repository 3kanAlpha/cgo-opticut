import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def process_image(img_rgb_pil: Image.Image, img_edit_pil: Image.Image):
  img_rgb = np.array(img_rgb_pil.convert("RGB")) / 255.0
  img_gray = np.array(img_rgb_pil.convert("L")) / 255.0
  img_edit = np.array(img_edit_pil.convert("L")) / 255.0
  img_hint = np.full(img_edit.shape, 0.5)
  #idx = (np.abs((img_gray-img_edit).sum(2)) > 1e-4)
  idx = (np.abs(img_gray - img_edit) > 1e-4)
  img_hint[idx] = img_edit[idx]

  # ヒントマスク（白→残す、黒→背景）
  white_mask = img_hint > 0.99
  black_mask = img_hint < 0.01

  # 輝度画像（Y）を使う
  img_yuv = color.rgb2yuv(img_rgb)
  Y = img_yuv[:, :, 0]

  w, h = Y.shape
  wpx = 1 # window size
  b = np.zeros( (w*h,) )
  # Sparse matrix
  row = []
  col = []
  dat = []

  for u in range(w):
      for v in range(h):
          i = v*w + u
          row.append(i)
          col.append(i)
          dat.append(1.)

          if white_mask[u, v]:
              row.append(i)
              col.append(i)
              dat.append(1.0)
              b[i] = 1.0
              continue
          elif black_mask[u, v]:
              row.append(i)
              col.append(i)
              dat.append(0.0)
              b[i] = 0.0
              continue

          umin = max(0,u-wpx)
          umax = min(w,u+wpx+1)
          vmin = max(0,v-wpx)
          vmax = min(h,v+wpx+1)
          patch = Y[umin:umax, vmin:vmax]
          mu_r = np.mean( patch )
          sigma_r = np.var( patch )
          sigma_r = max( sigma_r, 1e-6 )
          Yr = Y[u, v]

          # Go over neighbours
          N = []
          wrs = []
          for nu in range( umin, umax ):
              for nv in range( vmin, vmax ):
                  if (nu == u) and (nv == v):
                      continue
                  j = nv * w + nu
                  Ys = Y[nu, nv]
                  wrs.append(np.exp(-(Yr - Ys)**2 / (2 * sigma_r)))
                  N.append(j)

          wrs = np.array(wrs)
          wrs /= wrs.sum()

          for k,j in enumerate(N):
              row.append( i )
              col.append( j )
              dat.append( -wrs[k] )

  A = csr_matrix((dat, (row, col)), shape=(w*h, w*h))

  # 解く
  m = spsolve(A, b).reshape(h, w).transpose()

  # 2値マスクに変換（しきい値処理）
  mask_binary = (m > 0.3).astype(np.uint8)

  # 元のRGB画像をuint8に変換
  img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)

  # アルファチャンネルを作成 (マスクが0の部分が透明(0)、1の部分が不透明(255))
  alpha_channel = (mask_binary * 255).astype(np.uint8)

  # RGBとアルファチャンネルを結合してRGBA画像を作成
  rgba_image = np.dstack((img_rgb_uint8, alpha_channel))

  # 結果を PIL Image に変換して返す
  mask_binary_pil = Image.fromarray((mask_binary * 255).astype(np.uint8), mode='L')
  # RGBAモードで前景画像を生成
  foreground_image_pil = Image.fromarray(rgba_image, mode='RGBA')

  return mask_binary_pil, foreground_image_pil

def main():
  pass

if __name__ == '__main__':
  main()