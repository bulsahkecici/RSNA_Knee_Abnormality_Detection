from __future__ import annotations

import csv

import numpy as np
import pydicom

from rsna_knee.data.cache_build import build_cache
from rsna_knee.ontology import SERIES_ID_COL, STUDY_ID_COL


def _write_slice(path, study, series, instance, z):
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds = pydicom.dataset.FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study
    ds.SeriesInstanceUID = series
    ds.Modality = "MR"
    ds.InstanceNumber = instance
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, float(z)]
    ds.PixelSpacing = [0.5, 0.5]
    ds.Rows = 8
    ds.Columns = 8
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    arr = (np.arange(64, dtype=np.uint16).reshape(8, 8) + instance).copy()
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def test_cache_orders_geometry_and_quarantines_missing_study(tmp_path):
    study = "1.2.840.study"
    series = {
        "sag": ("Sagittal", "1"),
        "cor": ("Coronal", "1"),
    }
    rows = []
    for name, (plane, fluid) in series.items():
        sid = f"series-{name}"
        folder = tmp_path / "dicom" / study / sid
        folder.mkdir(parents=True)
        # filenames are reversed relative to IPP so lexicographic order is wrong
        for instance, z in ((3, 30.0), (1, 10.0), (2, 20.0)):
            _write_slice(folder / f"z{z}.dcm", study, sid, instance, z)
        rows.append(
            {
                STUDY_ID_COL: study,
                SERIES_ID_COL: sid,
                "Anatomical_Plane": plane,
                "Fluid_Sensitive": fluid,
                "Fat_Suppression": fluid,
            }
        )
    rows.append(
        {
            STUDY_ID_COL: "missing.study",
            SERIES_ID_COL: "series-missing",
            "Anatomical_Plane": "Sagittal",
            "Fluid_Sensitive": "1",
            "Fat_Suppression": "1",
        }
    )
    csv_path = tmp_path / "train_series.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = build_cache(
        series_csv=csv_path,
        dicom_root=tmp_path / "dicom",
        dest=tmp_path / "cache",
        size=28,
        n_centers=3,
    )
    assert "missing.study" in manifest["quarantine"]
    assert not (tmp_path / "cache" / "missing.study.npy").exists()
    meta = manifest["studies"][study]
    assert meta["shape"][0] == 3
    assert meta["shape"][1] == 3
    assert meta["shape"][2] == 3
    assert meta["dtype"] == "uint8"
    assert meta["order"] == "iop_ipp"
    assert meta["slot_mask"][2] == 0.0
    assert "sha256" in meta
    volume = np.load(tmp_path / "cache" / f"{study}.npy")
    assert volume.shape[2] == 3


def test_inference_masks_bad_series_but_never_caches_empty_study(tmp_path, monkeypatch):
    rows = [
        {STUDY_ID_COL: 'study', SERIES_ID_COL: 'bad', 'Anatomical_Plane': 'Sagittal', 'Fluid_Sensitive': '1'},
        {STUDY_ID_COL: 'study', SERIES_ID_COL: 'good', 'Anatomical_Plane': 'Coronal', 'Fluid_Sensitive': '1'},
    ]
    series_csv = tmp_path/'series.csv'
    with series_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path/'dicom'/'study').mkdir(parents=True)
    monkeypatch.setattr('rsna_knee.data.cache_build._load_slices',
                        lambda p: ([], 'codec_or_decode:test') if p.name == 'bad' else ([(None, None)], None))
    monkeypatch.setattr('rsna_knee.data.cache_build.series_to_centers',
                        lambda pairs, **kwargs: (np.ones((3, 3, 28, 28), dtype=np.float32), 'test'))
    result = build_cache(series_csv=series_csv, dicom_root=tmp_path/'dicom', dest=tmp_path/'partial',
                         strict_series=False, size=28)
    assert result['studies']['study']['slot_mask'] == [0.0, 1.0, 0.0]
    assert result['studies']['study']['series_errors'][0]['series'] == 'bad'
    strict = build_cache(series_csv=series_csv, dicom_root=tmp_path/'dicom', dest=tmp_path/'strict', size=28)
    assert 'study' in strict['quarantine']
    monkeypatch.setattr('rsna_knee.data.cache_build._load_slices', lambda p: ([], 'codec_or_decode:test'))
    empty = build_cache(series_csv=series_csv, dicom_root=tmp_path/'dicom', dest=tmp_path/'empty',
                        strict_series=False, size=28)
    assert 'study' in empty['quarantine']
    assert not (tmp_path/'empty'/'study.npy').exists()
