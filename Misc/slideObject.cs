// 21BAI1225 - Swarab Raul

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class slideObject : MonoBehaviour
{
    public Slider sizeSlider;
    public GameObject objectToResize;
    private Vector3 orginalScale;

    // Start is called before the first frame update
    void Start()
    {
        orginalScale = objectToResize.transform.localScale;
        sizeSlider.onValueChanged.AddListener(AdjustSize);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void AdjustSize(float value) {
        objectToResize.transform.localScale = orginalScale * value;
    }
}
