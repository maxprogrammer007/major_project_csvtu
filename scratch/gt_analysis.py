data = [
    # (imgid, gt_nonempty, raw, post, raw_ov)
    (629558, 34,  345, 12,  33766),
    (634445, 136, 152, 122, 34),
    (670433, 53,  59,  53,  28),
    (677385, 137, 285, 122, 45),
    (729453, 192, 245, 185, 40),
]

print('=== Catastrophic Table Analysis vs Ground Truth ===\n')
hdr = f"{'imgid':>8} {'gt_ne':>7} {'raw':>7} {'post':>6} {'raw_ov':>8}  {'raw_err%':>9}  {'post_err%':>10}"
print(hdr)
print('-'*60)

for imgid, gt, raw, post, ov in data:
    raw_err  = 100 * (raw - gt)  / gt
    post_err = 100 * (post - gt) / gt
    print(f'{imgid:>8} {gt:>7} {raw:>7} {post:>6} {ov:>8}  {raw_err:>+9.0f}%  {post_err:>+10.0f}%')

print()
print('Ordinary comparison tables:')
ord_data = [
    (622581, 33, 35, 33, 9, 0),
    (624025, 50, 51, 50, 18, 0),
    (639579, 61, 76, 61, 0, 1),
    (722429, 46, 40, 29, 21, 1),
]
for imgid, gt, raw, post, raw_ov, post_ov in ord_data:
    raw_err  = 100 * (raw - gt)  / gt
    post_err = 100 * (post - gt) / gt
    print(f'{imgid:>8} {gt:>7} {raw:>7} {post:>6} {raw_ov:>8}  {raw_err:>+9.0f}%  {post_err:>+10.0f}%  post_ov={post_ov}')
