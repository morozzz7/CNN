"""
Ищет в датасете файлы, которые либо не открываются, либо декодируются
аномально долго (что в py-spy dump выглядит как "завис внутри to_tensor").

Каждый файл проверяется в отдельном воркере с таймаутом — если воркер
завис на конкретном файле, мы это увидим и не будем ждать его вечно,
а просто пометим файл как подозрительный и продолжим со следующими.

Запуск:
    python scan_dataset_for_bad_images.py flickr30k_images

На выходе — bad_images.txt со списком путей, которые стоит убрать/перекачать
перед следующим запуском обучения.
"""
import os
import sys
import time
import multiprocessing as mp

from PIL import Image
from torchvision import transforms

CHECK_TIMEOUT_SEC = 10   # если ОДНА картинка декодируется дольше — подозрительна
PROCESSES = max(1, (os.cpu_count() or 4) - 1)


def _check_one(path):
    """Выполняется в воркер-процессе: открыть + полностью декодировать +
    прогнать через тот же to_tensor, что и в реальном пайплайне."""
    img = Image.open(path).convert('RGB')      # .convert() форсирует полный декодинг
    _ = transforms.functional.to_tensor(img)    # та самая строка, где завис процесс
    return True


def main(image_dir):
    paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    print(f"Файлов для проверки: {len(paths)} (процессов: {PROCESSES})")

    bad = []
    t0 = time.time()

    with mp.Pool(processes=PROCESSES) as pool:
        pending = {pool.apply_async(_check_one, (p,)): p for p in paths}
        for i, (future, path) in enumerate(pending.items()):
            try:
                future.get(timeout=CHECK_TIMEOUT_SEC)
            except mp.TimeoutError:
                print(f"[HANG]  {path} — не ответил за {CHECK_TIMEOUT_SEC}s")
                bad.append((path, 'timeout'))
            except Exception as e:
                print(f"[ERROR] {path} — {type(e).__name__}: {e}")
                bad.append((path, repr(e)))

            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                print(f"  ...{i + 1}/{len(paths)} проверено, {elapsed:.0f}s, "
                      f"найдено проблемных: {len(bad)}")

    print(f"\nГотово за {time.time() - t0:.0f}s. Проблемных файлов: {len(bad)}")
    with open('bad_images.txt', 'w') as f:
        for path, reason in bad:
            f.write(f"{path}\t{reason}\n")
    print("Список сохранён в bad_images.txt")


if __name__ == '__main__':
    image_dir = sys.argv[1] if len(sys.argv) > 1 else 'flickr30k_images'
    main(image_dir)
