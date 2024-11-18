// 21BAI1225 - Swarab Raul

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class SetColor : MonoBehaviour
{
    SpriteRenderer sprite;
    public Color newColor;
    public Button B1, B2;

    // Start is called before the first frame update
    void Start()
    {
        Button btn1 = B1.GetComponent<Button>();
        btn1.onClick.AddListener(B1Click);
        Button btn2 = B2.GetComponent<Button>();
        btn2.onClick.AddListener(B2Click);
    }

    void B1Click()
    {
        Debug.Log("Red Color");
        sprite = GetComponent<SpriteRenderer>();
        sprite.color = Color.red;
    }
    void B2Click()
    {
        Debug.Log("Green Color");
        sprite = GetComponent<SpriteRenderer>();
        sprite.color = Color.green;
    }
}
